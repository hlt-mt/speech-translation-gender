import logging
import sys
from typing import Dict, Optional, Tuple, Any

import numpy as np
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.init as init

from src.probes import BaseProbe, register_probe
from src.dataset import CustomDataset, collate_fn


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


@register_probe("feedforward")
class FeedForwardProbe(BaseProbe, nn.Module):
    def __init__(
            self,
            max_iter: int = 1000,
            tol: float = 0.0001,
            learning_rate: float = 0.001,
            num_layers: int = 2,
            embedding_dim: int = 1024,
            dropout: float = 0.1,
            seed: Optional[int] = None,
            *args,
            **kwargs) -> None:
        BaseProbe.__init__(
            self, max_iter=max_iter, tol=tol, seed=seed)
        nn.Module.__init__(self)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = learning_rate
        # Initialize model
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(embedding_dim, 2))  # Output layer for binary classification
        self.classifier = nn.Sequential(*layers)  # Wrap in Sequential
        self._initialize_weights()  # Initialize weights with Xavier

        LOGGER.info(f"Learning Rate: {learning_rate}")
        LOGGER.info(f"Number of linear layers: {num_layers}, "
                    f"Embedding dimension: {embedding_dim}, "
                    f"Dropout: {dropout}")

        self.to(self.device)

    def _initialize_weights(self) -> None:
        """
        Initializes weights of linear layers using Xavier initialization.
        """
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier (Glorot) initialization
                if layer.bias is not None:
                    init.zeros_(layer.bias)  # Optional: initialize biases to zero

    def forward(
            self,
            x: torch.Tensor,
            *args,
            **kwargs) -> torch.Tensor:
        logits = self.classifier(x)
        return logits

    def get_linear_weights(self):
        layer = self.classifier[-1]
        if isinstance(layer, nn.Linear):
            return layer.weight.data.cpu().numpy(), layer.bias.data.cpu().numpy()
        else:
            raise ValueError(f"The last layer is not a linear layer.")

    def do_training(
            self,
            X_train: np.ndarray,  # 2d embeddings, the first dimension represents the sample
            Y_train: np.ndarray,
            X_val: np.ndarray = None,
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
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
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

                logits = self(padded_sequences)
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

                        logits = self(padded_sequences)
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
        train_preds, _, train_report = self.do_evaluation(
            X_train, Y_train, batch_size=batch_size)
        training_acc = train_report["accuracy"]

        # Calculate final validation accuracy if validation data is available
        if X_val is not None and Y_val is not None:
            val_preds, _, val_report = self.do_evaluation(
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
            X: np.ndarray,  # 2d embeddings, the first dimension represents the sample
            Y: np.ndarray,
            batch_size: int = 1000,
            *args,
            **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict, Optional[Any]]:

        dataset = CustomDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        all_preds = []
        all_female_probs = []
        self.eval()

        with torch.no_grad():
            for padded_sequences, pad_mask, batch_labels in dataloader:
                padded_sequences = padded_sequences.to(self.device)
                logits = self(padded_sequences)

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

        return all_preds, all_female_probs, report, None  # None in place of the attention weights

    def load_model(self, model_path: str) -> None:
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.load_state_dict(state_dict)
            LOGGER.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            LOGGER.error(f"Failed to load model from {model_path}: {e}")
            raise

    def save_model(self, save_path: str) -> None:
        torch.save(self.state_dict(), save_path + ".pt")
