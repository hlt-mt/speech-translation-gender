#!/usr/bin/env python3
from typing import List, Tuple
import yaml
import argparse
import random

import pandas as pd


def map_talk_id(
        df: pd.DataFrame,
        yaml_data: List[dict],
        source_texts: List[str],
        target_texts: List[str]) -> Tuple[List[dict], List[dict]]:
    """
    Maps TALK-ID to SPEAKER-ID.
    """
    talk_to_speaker = dict(zip(df["TALK-ID"].astype(str), zip(df["SPEAKER-ID"], df["TED-PRONOUN"])))
    print(talk_to_speaker)
    new_yaml_M = []
    new_yaml_F = []
    for idx, entry in enumerate(yaml_data):
        wav_file = entry['wav']
        talk_id = wav_file.split('_')[1].split('.')[0]  # Extract TALK-ID from wav filename
        # Replace the placeholder speaker_id with the meaningful SPEAKER-ID
        entry['speaker_id'] = talk_to_speaker[talk_id][0]
        entry['src_text'] = source_texts[idx].strip()
        entry['tgt_text'] = target_texts[idx].strip()
        if talk_to_speaker[talk_id][1] == "She":
            new_yaml_F.append(entry)
        elif talk_to_speaker[talk_id][1] == "He":
            new_yaml_M.append(entry)
        else:
            continue
    print(f"Total n. of entries for M: {len(new_yaml_M)}")
    print(f"Total n. of entries for F: {len(new_yaml_F)}")

    return new_yaml_M, new_yaml_F


def filter_yaml_and_text(
        yaml_path: str,
        df: pd.DataFrame,
        source_txt_path: str,
        target_txt_path: str,
        output_yaml_path: str) -> None:
    with (open(yaml_path, 'r') as f, \
            open(source_txt_path, 'r', encoding='utf-8') as f_src, \
            open(target_txt_path, 'r', encoding='utf-8') as f_tgt):
        yaml_data = yaml.safe_load(f)
        source_texts = f_src.readlines()
        target_texts = f_tgt.readlines()
    new_yaml_M, new_yaml_F = map_talk_id(df, yaml_data, source_texts, target_texts)

    print("Saving...")
    with open(output_yaml_path + "_she.yaml", 'w') as f_yaml:
        yaml.safe_dump(new_yaml_F, f_yaml, encoding='utf-8', allow_unicode=True)
    with open(output_yaml_path + "_he.yaml", 'w') as m_yaml:
        yaml.safe_dump(new_yaml_M, m_yaml, encoding='utf-8', allow_unicode=True)


def main(
        tsv_path: str,
        yaml_path: str,
        source_txt_path: str,
        target_txt_path: str,
        output_yaml_file: str) -> None:
    df = pd.read_csv(tsv_path, sep="\t")
    filter_yaml_and_text(
        yaml_path,
        df,
        source_txt_path,
        target_txt_path,
        output_yaml_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter TSV and YAML files based on gender and write corresponding text files.")
    parser.add_argument(
        '--tsv-path',
        type=str,
        required=True,
        help='Path to the MuST-Speaker TSV file.')
    parser.add_argument(
        '--yaml-path',
        type=str,
        required=True,
        help='Path to the input YAML file.')
    parser.add_argument(
        '--source-txt-path',
        type=str,
        required=True,
        help='Path to the source text file.')
    parser.add_argument(
        '--target-txt-path',
        type=str,
        required=True,
        help='Path to the target text file.')
    parser.add_argument(
        '--output-yaml-path',
        type=str,
        required=True,
        help='Path to the output yaml file where data will be saved.')
    args = parser.parse_args()

    random.seed(42)
    main(
        args.tsv_path,
        args.yaml_path,
        args.source_txt_path,
        args.target_txt_path,
        args.output_yaml_path)
