#!/usr/bin/env python3
# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import argparse
import logging
import sys
from typing import Optional

import pandas as pd

from src.loaders import get_loader
from src.probes import get_probe
from src.utils import set_seed


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


def main(
        train_tsv_path: str,
        train_embeddings_path: str,
        save_probe: str,
        probe_type: str,
        level: str,
        max_iter: int,
        batch_size: int,
        update_frequency: int,
        tol: float,
        lr: float,
        dropout: float,
        dropout_att: float,
        l1_lambda: float,
        l2_lambda: float,
        num_layers: int,
        num_heads: int,
        attention_type: str,
        early_stopping_rounds: int,
        seed: int,
        classes_balance: bool,
        position_index: Optional[int] = None,
        relative_frames: Optional[int] = None,
        val_tsv_path: Optional[str] = None,
        val_embeddings_path: Optional[str] = None) -> None:
    """
    Train and save a prober to classify gender attribute.
    """
    set_seed(seed)
    LOGGER.info(f"Set seed to {seed}")

    LOGGER.info("Loading data...")
    loader = get_loader(level)

    train_df = pd.read_csv(train_tsv_path, sep="\t")
    train_loader = loader(train_embeddings_path, train_df)
    x_train, y_train = train_loader(
        classes_balance=classes_balance,
        position_index=position_index,
        relative_frames=relative_frames)
    embedding_dim_train = x_train.shape[1] if not isinstance(x_train, list) else x_train[0].shape[1]
    LOGGER.info(f"Training data have {len(x_train)} samples, with embedding dimension of {embedding_dim_train}")
    assert len(x_train) == len(y_train), "Length of x_train and y_train does not match"

    x_val, y_val = None, None
    if val_tsv_path is not None and val_embeddings_path is not None:
        val_df = pd.read_csv(val_tsv_path, sep="\t")
        val_loader = loader(val_embeddings_path, val_df)
        x_val, y_val = val_loader(
            classes_balance=classes_balance,
            position_index=position_index,
            relative_frames=relative_frames)
        embedding_dim_val = x_val.shape[1] if not isinstance(x_val, list) else x_val[0].shape[1]
        LOGGER.info(f"Training data have {len(x_val)} samples, with embedding dimension of {embedding_dim_val}")
        assert len(x_val) == len(y_val), "Length of x_val and y_val does not match"

        assert embedding_dim_val == embedding_dim_train, "Embedding size of train and val does not match"

    # Initializing the probe
    probe_model = get_probe(probe_type)
    probe = probe_model(
        max_iter=max_iter,
        tol=tol,
        learning_rate=lr,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dim=x_train[0].shape[-1],
        dropout=dropout,
        seed=seed,
        dropout_att=dropout_att,
        attention_type=attention_type)
    LOGGER.info(f"Probe initialized with {probe}")

    # Training
    LOGGER.info(
        f"{max_iter} maximum iterations with early stopping after {early_stopping_rounds} "
        f"iterations if loss does not improve by {tol}.")
    LOGGER.info(f"Batch size: {batch_size}, update frequency: {update_frequency}, learning rate: {lr}")

    LOGGER.info("Training the probe...")
    probe.do_training(
        X_train=x_train,
        Y_train=y_train,
        X_val=x_val,
        Y_val=y_val,
        save=save_probe,
        batch_size=batch_size,
        early_stopping_rounds=early_stopping_rounds,
        update_frequency=update_frequency,
        l1_lambda=l1_lambda,
        l2_lambda=l2_lambda)
    LOGGER.info(f"Probe(s) saved to {save_probe}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate prober for gender attribute.")
    parser.add_argument(
        '--dataframe-train',
        type=str,
        help='Path to the file containing the training data.')
    parser.add_argument(
        '--embeddings-train',
        type=str,
        help='Path to the tensor file where the training embeddings are stored.')
    parser.add_argument(
        '--dataframe-val',
        type=str,
        default=None,
        help='Path to the file containing the validation data. '
             'By default, validation is not used')
    parser.add_argument(
        '--embeddings-val',
        type=str,
        default=None,
        help='Path to the tensor file where the validation embeddings are stored. '
             'By default, validation is not used.')
    parser.add_argument(
        '--probe',
        type=str,
        choices=["logistic", "feedforward", "attention"],
        help='Type of probe to be used for training.')
    parser.add_argument(
        '--level',
        type=str,
        choices=["position", "average", "sequence", "max"],
        help='Level on which training the probe.')
    parser.add_argument(
        '--position-index',
        type=int,
        default=0,
        help='Index of the position in the embeddings to select. If --relative-frames is specified, '
             'this is treated as a relative index (0-based). Defaults to 0, meaning the first position.')
    parser.add_argument(
        '--relative-frames',
        type=int,
        default=None,
        help='Specifies the number of frames to sample embeddings from. If set, --position-index is '
             'interpreted as a relative index within the sampled frames. Defaults to None, meaning '
             '--position-index is treated as an absolute index.')
    parser.add_argument(
        '--save-probe',
        type=str,
        help='Path to save the probe.')
    parser.add_argument(
        '--classes-balance',
        action='store_true',
        help='Use this flag to enable balancing of classes at frame level. Default is true.')
    parser.add_argument(
        '--max-iter',
        type=int,
        default=2000,
        help='Number of maximum iterations performed during fitting. Default dis 2000.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Size of batch used for training. Default is 10000.')
    parser.add_argument(
        '--update-frequency',
        type=int,
        default=1,
        help='Number of batches to accumulate gradients before updating model weights. '
             'Default is 1.')
    parser.add_argument(
        '--tol',
        type=float,
        default=0.0001,
        help='Stopping criterion for the training, which will stop when '
             '(loss > best_loss - tol) for 5 consecutive epochs. Default is 0.0001.')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout probability for nonlinear probes. Default is 0.1.')
    parser.add_argument(
        '--dropout-att',
        type=float,
        default=0.0,
        help='Dropout probability for attention. Default is 0.0.')
    parser.add_argument(
        '--num-layers',
        type=int,
        default=1,
        help='Number of layers in nonlinear probes. Default is 1.')
    parser.add_argument(
        '--num-heads',
        type=int,
        default=1,
        help='Number of heads for the attention pooling. Default is 1.')
    parser.add_argument(
        '--attention-type',
        type=str,
        default='pooling',
        choices=['pooling', 'scaled_dot', 'custom'],
        help='Type of attention to be used. Default is "pooling".')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate to be used during training. '
             'If set to 0 with logistic probe, the default scikit-learn optimal lr is used.'
             'Default is 0.001.')
    parser.add_argument(
        '--l1-lambda',
        type=float,
        default=0.0,
        help='L1 regularization strength. Default is 0.0 (no regularization).')
    parser.add_argument(
        '--l2-lambda',
        type=float,
        default=0.0,
        help='L2 regularization strength. Default is 0.0 (no regularization).')
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=5,
        help='Rounds with no loss improvements after which the training is stopped. '
             'Default is 5.')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed.')
    args = parser.parse_args()

    if args.probe == "attention":
        assert args.level == "sequence", \
            "attention-based probes only supports sequence level."
    else:
        assert args.level != "sequence", \
            "Sequence level is only supported by attention-based probes."

    main(
        args.dataframe_train,
        args.embeddings_train,
        args.save_probe,
        args.probe,
        args.level,
        args.max_iter,
        args.batch_size,
        args.update_frequency,
        args.tol,
        args.learning_rate,
        args.dropout,
        args.dropout_att,
        args.l1_lambda,
        args.l2_lambda,
        args.num_layers,
        args.num_heads,
        args.attention_type,
        args.early_stopping,
        args.seed,
        args.classes_balance,
        args.position_index,
        args.relative_frames,
        args.dataframe_val,
        args.embeddings_val)
