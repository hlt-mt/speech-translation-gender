import csv
import sys
import argparse
import logging

import pandas as pd

import torch
import torch.multiprocessing

from src.transcriber import BaseTranscriber


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)

TARGET_SAMPLING_RATE = 16000


def main(
        tsv_dataset: str,
        lang: str,
        model: str,
        output_file: str,
        max_seq_len: int,
        batch_size: int,
        num_workers: int) -> None:
    df = pd.read_csv(tsv_dataset, sep="\t")
    file_paths = df["AUDIO"].values

    # Loading model
    transcriber = BaseTranscriber(
        model_name_or_path=model,
        tgt_lang=lang,
        device="cuda",
        torch_dtype=torch.bfloat16)
    LOGGER.info(f"Transcriber loaded. Model {model} has been chosen.")

    # Get transcriptions/translations
    LOGGER.info(f"Loading {len(df['AUDIO'])} sample data and getting transcriptions/translations.")
    transcriptions = transcriber(
        file_paths=file_paths,
        sampling_rate=TARGET_SAMPLING_RATE,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_seq_len)

    # Save outputs
    result = {
        "id": df["ID"],
        "SRC": df["SRC"],
        "REF": df["REF"],
        "OUTPUT": transcriptions,
        "SPEAKER": df["SPEAKER"],
        "GENDER": df["GENDER"],
        "CATEGORY": df["CATEGORY"],
        "GENDERTERMS": df["GENDERTERMS"]}
    results = pd.DataFrame(result)
    results.to_csv(
        output_file, encoding="utf-8", sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    LOGGER.info(f"Results written to {output_file}")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv-path', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--max-seq-len', type=int, default=None)
    parser.add_argument('--batch-size', type=int, required=False, default=1)
    parser.add_argument('--num-workers', type=int, required=False, default=4)
    args = parser.parse_args()
    main(
        args.tsv_path,
        args.lang,
        args.model_name,
        args.output_file,
        args.max_seq_len,
        args.batch_size,
        args.num_workers)


if __name__ == "__main__":
    _main()
