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
import csv
import logging
import pickle
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.loaders import get_loader
from src.probes import get_probe


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


def main(
        tsv_path: str,
        embeddings_path: str,
        pretrained_probe: str,
        probe_type: str,
        level: str,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        attention_type: str,
        position_index: Optional[int] = None,
        relative_frames: Optional[int] = None,
        output_tsv: Optional[str] = None,
        save_attention_weights: Optional[str] = None,
        save_linear_weights: Optional[str] = None) -> None:
    """
    Load a pretrained probe to evaluate gender attribute.
    """
    LOGGER.info("Preparing data...")
    _loader = get_loader(level)
    df = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE, escapechar='\\')
    loader = _loader(embeddings_path, df)

    x, y = loader(
        classes_balance=False,
        position_index=position_index,
        relative_frames=relative_frames)
    LOGGER.info(f"Embeddings loaded and processed")

    _probe = get_probe(probe_type)
    probe = _probe(
        model=pretrained_probe,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dim=x[0].shape[-1],
        attention_type=attention_type)
    probe.load_model(pretrained_probe)
    LOGGER.info(f"Using probe {pretrained_probe}")

    y_pred, female_probs, classification_report, attention_weights = probe.do_evaluation(
        x, y, batch_size=batch_size, return_attention=True)
    LOGGER.info("Overall Evaluation Report:")
    for label, metrics in classification_report.items():
        if isinstance(metrics, dict):  # Per-class metrics (e.g., precision, recall, f1-score, support)
            formatted_metrics = {
                key: (f"{value:.4f}" if isinstance(value, float) else value)
                for key, value in metrics.items()}
            LOGGER.info(f"{label}: {formatted_metrics}")
        else:
            # Handle overall metrics like accuracy (not dictionaries)
            LOGGER.info(f"{label}: {metrics:.4f}")

    if save_linear_weights is not None:
        linear_weights, bias = probe.get_linear_weights()
        np.save(save_linear_weights, linear_weights)
        LOGGER.info(f"Linear weights saved to: {save_linear_weights}")
    if save_attention_weights is not None:
        with open(save_attention_weights, "wb") as f:
            pickle.dump(attention_weights, f)
        LOGGER.info(f"Attention weights saved to: {save_attention_weights}")

    if output_tsv is not None:
        result_df = df.copy()
        result_df["Y_PREDS"] = y_pred
        result_df["F_PROBS"] = female_probs
        result_df.to_csv(output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate prober for gender attribute.")
    parser.add_argument(
        '--dataframe',
        type=str,
        help='Path to the file containing the data.')
    parser.add_argument(
        '--embeddings',
        type=str,
        help='Path to the tensor file where the embeddings are stored.')
    parser.add_argument(
        '--probe',
        type=str,
        choices=["logistic", "feedforward", "attention"],
        help='Type of probe to be used for training: logistic or nonlinear. '
             'If --level is position-level, just the path until _pos should be provided.')
    parser.add_argument(
        '--pretrained-probe',
        type=str,
        help='Path to the pickle file where the pretrained probe is saved')
    parser.add_argument(
        '--output-tsv',
        type=str,
        default=None,
        help='Path to tsv file where output will be saved.')
    parser.add_argument(
     '--batch-size',
        type=int,
        default=10000,
        help='Batch size used for inference.')
    parser.add_argument(
        "--save-attention-weights",
        type=str,
        default=None,
        help="Path to the npy file where attention weights are saved.")
    parser.add_argument(
        "--save-linear-weights",
        type=str,
        default=None,
        help="Path to the npy file where weights of the last linear layer are saved.")
    parser.add_argument(
        '--level',
        type=str,
        choices=["position", "frame", "average", "max", "sequence"],
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
        '--num-layers',
        type=int,
        default=2,
        help='Number of layers for nonlinear classifiers.')
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
    args = parser.parse_args()

    if args.probe == "attention":
        assert args.level == "sequence", \
            "attention-based probes only supports sequence level."
    else:
        assert args.level != "sequence", \
            "Sequence level is only supported by attention-based probes."

    main(
        args.dataframe,
        args.embeddings,
        args.pretrained_probe,
        args.probe,
        args.level,
        args.batch_size,
        args.num_layers,
        args.num_heads,
        args.attention_type,
        args.position_index,
        args.relative_frames,
        args.output_tsv,
        args.save_attention_weights,
        args.save_linear_weights)
