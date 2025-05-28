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

import h5py
import pandas as pd

import torch
from tqdm import tqdm

from src.transcriber import BaseTranscriber


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)

TARGET_SAMPLING_RATE = 16000
BATCH_SIZE = 1

SWOT_MODEL_NAME = "Swot"
MUSTC_MODEL_NAME = "mustc"
WHISPER_MODEL_NAME = "whisper"
SEAMLESS_MODEL_NAME = "seamless"
HUBERT_MODEL_NAME = "hubert"


def main(
        model: str,
        lang: str,
        tsv_dataset: str,
        output_file: str,
        layer: str,
        max_seq_len: int,
        num_workers: int,
        show_progress_bar: bool = True) -> None:
    """
    Extract and saves embeddings from the encoder.
    """
    # Loading data
    df = pd.read_csv(tsv_dataset, sep="\t")
    file_paths = df["AUDIO"].values

    # Loading model
    transcriber = BaseTranscriber(
        model_name_or_path=model,
        tgt_lang=lang,  # actually not used for encoding
        device="cuda",
        torch_dtype=torch.bfloat16)
    LOGGER.info(f"Transcriber loaded. Model {model} has been chosen.")

    # Extract embeddings
    LOGGER.info(f"Loading {len(df['AUDIO'])} sample data and extracting embeddings.")
    loader = transcriber.build_loader(
        file_paths=file_paths,
        sampling_rate=TARGET_SAMPLING_RATE,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        max_length=max_seq_len)

    with h5py.File(output_file, "a") as h5f:
        for idx, batch in tqdm(
                enumerate(loader), desc="Batch", disable=not show_progress_bar, total=len(loader)):
            batch = {
                k: v.to(dtype=transcriber.model_type, device=transcriber.device)
                if k != "attention_mask" else v.to(transcriber.device)
                for k, v in batch.items()}

            if SEAMLESS_MODEL_NAME in model:
                encoder = transcriber.model.speech_encoder
                out = encoder(**batch, output_hidden_states=True)
                if layer == "pre_adapter":
                    last_hs = out.hidden_states[-1]
                else:
                    last_hs = out.last_hidden_state

            elif SWOT_MODEL_NAME in model:
                encoder = transcriber.zeroswot_encoder
                if layer == "pre_adapter":
                    out = encoder.wav2vec2.wav2vec2(**batch, output_hidden_states=False)
                    last_hs = out.last_hidden_state
                else:
                    out = None
                    last_hs, _ = encoder(**batch)

            elif WHISPER_MODEL_NAME in model:
                encoder = transcriber.model.model.encoder
                out = encoder(**batch, output_hidden_states=True)
                last_hs = out.hidden_states[-1]

            elif HUBERT_MODEL_NAME in model:
                hubert_model = transcriber.model.hubert
                out = hubert_model(**batch)
                last_hs = out.last_hidden_state

            else:
                encoder = transcriber.model.model.encoder
                out = encoder(**batch, output_hidden_states=True)
                last_hs = out.hidden_states[-1]

            del out  # free up memory

            # last_hs has shape (bs, seq_len, hsize)
            if last_hs.dtype == torch.bfloat16:
                last_hs = last_hs.to(torch.float32)
            elif last_hs.dtype != torch.float32:
                LOGGER.info("Embeddings tensor not casted into float32.")
            last_hs = last_hs.cpu().detach().numpy()

            # Save embeddings
            h5f.create_dataset(str(idx), data=last_hs)

    LOGGER.info(f"Embeddings saved to {output_file}")
    LOGGER.info(f"Total trimmed audio samples: {loader.dataset.trimmed_data_counter}")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model to use for generation.")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language to generate.")
    parser.add_argument(
        "--tsv-path",
        type=str,
        required=True,
        help="Path to the tsv file containing data.")
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the directory where embeddings will be saved.")
    parser.add_argument(
        "--layer",
        type=str,
        choices=["pre_adapter", "post_adapter"],
        required=True,
        help="The layer from which the embeddings are extracted: post_adapter, pre_adapter.")
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=None,
        help="Maximum length in seconds of the sequence")
    parser.add_argument(
        "--num-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to parallelize the loading of the dataset.")
    args = parser.parse_args()

    main(
        args.model_name,
        args.lang,
        args.tsv_path,
        args.output_file,
        args.layer,
        args.max_seq_len,
        args.num_workers)


if __name__ == "__main__":
    _main()
