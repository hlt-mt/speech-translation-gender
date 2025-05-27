import logging
import sys
from typing import List, Optional, Tuple, Dict

import numpy as np
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.probes import collate_fn, CustomDataset, register_probe
from src.probes.feedforward import FeedForwardProbe


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

    def forward(
            self,
            x: torch.Tensor,
            pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class ScaledDotAttention(Attention):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(ScaledDotAttention, self).__init__(embedding_dim, num_heads)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, num_heads=self.num_heads, batch_first=True)
        # Identity tensor as queries
        self.query = torch.ones(1, embedding_dim, requires_grad=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.query = self.query.to(self.device)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, embedding_dim)
        # query: (num_heads, embedding_dim)
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)  # (batch_size, 1, embedding_dim)
        # pad_mask==0 for padding tokens
        attn_mask = ~pad_mask.bool()  # (batch_size, seq_len)

        attn_output, attn_weights = self.multihead_attn(  # linear transformations computed internally
            query=query, key=x, value=x, key_padding_mask=attn_mask, need_weights=True, average_attn_weights=False)
        # attn_output: (batch_size, 1, embedding_dim) -> (batch_size, embedding_dim)
        # attn_weights: (batch_size, num_heads, 1, seq_len)

        attn_output = attn_output.squeeze(1)  # (batch_size, embedding_dim)
        attn_weights = attn_weights.squeeze(2)  # (batch_size, num_heads, seq_len)

        return attn_output, attn_weights


class CustomMultiHeadAttention(Attention):
    def __init__(self, embedding_dim: int, num_heads: int, dropout_att: float = 0.0):
        super(CustomMultiHeadAttention, self).__init__(embedding_dim, num_heads)
        self.head_dim = self.embedding_dim // self.num_heads
        assert self.embedding_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.dropout_att = nn.Dropout(p=dropout_att)

        # Learnable 1D query vector for all heads
        self.Q = nn.Parameter(torch.empty(1, embedding_dim))
        nn.init.xavier_uniform_(self.Q)  # Xavier initialization for Q

        # Learnable weights for projecting keys
        self.k_proj = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(
            self, x: torch.Tensor, pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()

        # Project keys
        K = self.k_proj(x)  # (batch_size, seq_len, embedding_dim)

        # Expand and reshape Q
        Q = self.Q.expand(batch_size, -1).unsqueeze(1)  # (batch_size, 1, embedding_dim)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, 1, head_dim)

        # Reshape K and V for multi-head attention
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, 1, seq_len)

        # Apply padding mask
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        scores = scores.masked_fill(pad_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout_att(attn_weights)
        attn_output = torch.matmul(attn_weights_drop, V)  # (batch_size, 1, 1, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, self.embedding_dim)

        return attn_output, attn_weights


class PoolingAttention(Attention):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(PoolingAttention, self).__init__(embedding_dim, num_heads)
        # Learnable 1D query vector for each head
        self.query = nn.Parameter(torch.empty(num_heads, embedding_dim))
        nn.init.xavier_uniform_(self.query)  # Apply Xavier initialization

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, embedding_dim)
        # query: (num_heads, embedding_dim)
        attention_scores = torch.matmul(x, self.query.T)  # (batch_size, seq_len, num_heads)
        attention_scores = attention_scores.masked_fill(pad_mask.unsqueeze(-1) == 0, -1e9)  # Handle padding
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize attention scores
        attention_weights_t = attention_weights.permute(0, 2, 1)  # (batch_size, num_heads, seq_len)
        weighted_sum = torch.matmul(attention_weights_t, x)  # (batch_size, num_heads, embedding_dim)
        return weighted_sum.view(x.size(0), -1), attention_weights  # Flatten (batch_size, num_heads*embedding_dim)


@register_probe("attention")
class AttentionPoolingProbe(FeedForwardProbe):
    def __init__(
            self,
            max_iter: int = 1000,
            tol: float = 0.0001,
            learning_rate: float = 0.001,
            num_layers: int = 2,
            num_heads: int = 1,
            embedding_dim: int = 1024,
            dropout: float = 0.1,
            seed: Optional[int] = None,
            dropout_att: float = 0.0,
            attention_type: str = "pooling",
            *args,
            **kwargs) -> None:
        super().__init__(
            max_iter,
            tol,
            learning_rate,
            num_layers,
            num_heads*embedding_dim,
            dropout,
            seed,
            *args,
            **kwargs)
        if attention_type == "pooling":
            self.attention = PoolingAttention(embedding_dim, num_heads)
        elif attention_type == "scaled_dot":
            self.attention = ScaledDotAttention(embedding_dim, num_heads)
        elif attention_type == "custom":
            self.attention = CustomMultiHeadAttention(embedding_dim, num_heads, dropout_att)
        else:
            raise ValueError(f"Invalid attention type {attention_type}.")
        LOGGER.info(f"Attention type before linear layer(s): {attention_type}")

    def forward(
            self,
            x: torch.Tensor,
            pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_weights = self.attention(x, pad_mask=pad_mask)
        logits = super().forward(attention_output)  # (batch_size, 2)
        return logits, attention_weights

    def do_training(
            self,
            X_train: List[np.ndarray],  # List of 2d embeddings (seq_len, embed_dim), one per sample
            Y_train: np.ndarray,
            X_val: List[np.ndarray] = None,
            Y_val: np.ndarray = None,
            save: Optional[str] = None,
            batch_size: int = 10000,
            early_stopping_rounds: int = 5,
            update_frequency: int = 1,
            l1_lambda: float = 0.0,
            l2_lambda: float = 0.0,
            *args,
            **kwargs) -> None:

        train_dataset = CustomDataset(X_train, Y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        if X_val is not None and Y_val is not None:
            val_dataset = CustomDataset(X_val, Y_val)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        else:
            val_loader = None  # Validation set is not provided

        # Optimizer and Scheduler
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        no_improvement_count = 0
        for iteration in range(self.max_iter):
            total_train_loss = 0
            optimizer.zero_grad()  # Reset gradients
            self.train()  # Training mode

            # Training loop over batches
            for batch_idx, (padded_sequences, pad_mask, batch_labels) in enumerate(train_loader):
                padded_sequences = padded_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                pad_mask = pad_mask.to(self.device)

                logits, _ = self(padded_sequences, pad_mask)
                loss = criterion(logits, batch_labels)

                # Add L1 regularization
                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in self.classifier.parameters())
                    loss += l1_lambda * l1_norm

                # Add L2 regularization
                if l2_lambda > 0:
                    l2_norm = sum(p.pow(2).sum() for p in self.classifier.parameters())
                    loss += l2_lambda * l2_norm

                loss = loss / update_frequency  # Normalize for gradient accumulation
                loss.backward()  # Backpropagation

                if (batch_idx + 1) % update_frequency == 0:
                    optimizer.step()  # Update weights after accumulation
                    optimizer.zero_grad()  # Reset gradients after update

                total_train_loss += loss.item() * update_frequency  # Recover original loss

            avg_train_loss = total_train_loss / len(train_loader)

            if val_loader:  # If validation data is provided, evaluate on validation set
                self.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for padded_sequences, pad_mask, batch_labels in val_loader:
                        padded_sequences = padded_sequences.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        pad_mask = pad_mask.to(self.device)

                        logits, _ = self(padded_sequences, pad_mask)
                        val_loss = criterion(logits, batch_labels)

                        # Add L1 and L2 regularization for validation
                        if l1_lambda > 0:
                            l1_norm = sum(p.abs().sum() for p in self.classifier.parameters())
                            val_loss += l1_lambda * l1_norm
                        if l2_lambda > 0:
                            l2_norm = sum(p.pow(2).sum() for p in self.classifier.parameters())
                            val_loss += l2_lambda * l2_norm

                        total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                current_loss = avg_val_loss  # Use validation loss for early stopping
                scheduler.step(avg_val_loss)  # Step the scheduler based on validation loss

                if iteration == 0 or (iteration + 1) % 10 == 0:
                    LOGGER.info(
                        f"Iteration {iteration + 1}, "
                        f"Train Loss: {avg_train_loss}, "
                        f"Val Loss: {avg_val_loss}, "
                        f"Learning Rate: {optimizer.param_groups[0]['lr']}")

            else:  # If validation data is not provided, use training loss for early stopping
                current_loss = avg_train_loss
                scheduler.step(avg_train_loss)  # Step the scheduler based on training loss
                if iteration == 0 or (iteration + 1) % 10 == 0:
                    LOGGER.info(
                        f"Iteration {iteration + 1}, "
                        f"Train Loss: {avg_train_loss} (No validation set), "
                        f"Learning Rate: {optimizer.param_groups[0]['lr']}")

            # Early stopping based on the current loss (either validation or training)
            best_loss, no_improvement_count = self.early_stopping_check(
                current_loss, best_loss, no_improvement_count, early_stopping_rounds)

            if no_improvement_count >= early_stopping_rounds:
                break
            if save and current_loss == best_loss:
                self.save_model(save)

        # Calculate final training accuracy
        train_preds, _, train_report, _ = self.do_evaluation(
            X_train, Y_train, batch_size=batch_size)
        training_acc = train_report["accuracy"]

        # Calculate final validation accuracy if validation data is available
        if X_val is not None and Y_val is not None:
            val_preds, _, val_report, _ = self.do_evaluation(
                X_val, Y_val, batch_size=batch_size)
            validation_acc = val_report["accuracy"]
        else:
            validation_acc = None

        LOGGER.info(
            f"Training complete: Iterations = {iteration + 1}, "
            f"Training Accuracy = {training_acc:.4f}, "
            f"Validation Accuracy = {validation_acc if validation_acc is not None else 'N/A'}")

    def do_evaluation(
            self,
            X: List[np.ndarray],  # List of 2d embeddings (seq_len, embed_dim), one per sample
            Y: np.ndarray,
            batch_size: int = 1000,
            return_attention: bool = False,
            *args,
            **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict, Optional[List[np.ndarray]]]:

        dataset = CustomDataset(X, Y)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn)

        all_preds = []
        all_female_probs = []
        all_attention_weights = []
        self.eval()

        with torch.no_grad():
            for padded_sequences, pad_mask, batch_labels in dataloader:
                padded_sequences = padded_sequences.to(self.device)
                pad_mask = pad_mask.to(self.device)

                logits, attention_weights = self(padded_sequences, pad_mask)
                all_attention_weights.append(attention_weights.cpu().numpy())

                # y_pred = torch.argmax(logits, dim=1).cpu()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_female = probs[:, 1]
                all_female_probs.append(probs_female)

                y_pred = np.argmax(probs, axis=1)
                all_preds.append(y_pred)

        all_preds = np.concatenate(all_preds)
        all_female_probs = np.concatenate(all_female_probs)
        report = classification_report(
            Y,
            all_preds,
            labels=list(self.GENDER_2_ID.values()),
            target_names=list(self.GENDER_2_ID.keys()),
            output_dict=True)

        if return_attention:
            return all_preds, all_female_probs, report, all_attention_weights
        return all_preds, all_female_probs, report, None
