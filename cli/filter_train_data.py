#!/usr/bin/env python3
import os
from typing import Tuple, List
import yaml
import argparse
import random

import pandas as pd


def filter_tsv(
        tsv_path: str, no_speakers_list: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(tsv_path, sep="\t")
    filtered_data = df[(df['MUST-C'] == 'train') & (~df['SPEAKER-ID'].isin(no_speakers_list))]
    df_She = filtered_data[filtered_data['TED-PRONOUN'] == 'She']
    df_He = filtered_data[filtered_data['TED-PRONOUN'] == 'He']
    print(f"N. Talks for F: {len(df_She)}")
    print(f"N. Talks for M: {len(df_He)}")
    return df_She, df_He


def map_and_split_by_talk_id(
        df: pd.DataFrame,
        yaml_data: List[dict],
        source_texts: List[str],
        target_texts: List[str],
        split_ratio: float = 0.8) -> Tuple[List[dict], List[dict]]:
    """
    Maps TALK-ID to SPEAKER-ID, filters yaml_data by TALK-ID from df,
    and splits it into two disjoint sections based on SPEAKER-ID.
    """
    # Create a mapping of TALK-ID to SPEAKER-ID from the TSV file (df)
    talk_to_speaker = dict(zip(df["TALK-ID"].astype(str), df["SPEAKER-ID"]))

    # Filter yaml_data and map TALK-ID to speakers' names
    filtered_yaml = []
    for idx, entry in enumerate(yaml_data):
        wav_file = entry['wav']
        talk_id = wav_file.split('_')[1].split('.')[0]  # Extract TALK-ID from wav filename
        if talk_id in talk_to_speaker:
            # Replace the placeholder speaker_id with the meaningful SPEAKER-ID
            entry['speaker_id'] = talk_to_speaker[talk_id]
            entry['src_text'] = source_texts[idx].strip()
            entry['tgt_text'] = target_texts[idx].strip()
            filtered_yaml.append(entry)

    # Group filtered data by SPEAKER-ID
    speaker_to_data = {}
    for entry in filtered_yaml:
        speaker = entry["speaker_id"]
        if speaker not in speaker_to_data:
            speaker_to_data[speaker] = []
        speaker_to_data[speaker].append(entry)

    # Split speakers into two disjoint groups based on the split ratio
    speakers = list(speaker_to_data.keys())
    random.shuffle(speakers)
    split_point = int(len(speakers) * split_ratio)
    speakers_group1 = speakers[:split_point]
    speakers_group2 = speakers[split_point:]
    print(f"N. Speakers for the first set: {len(speakers_group1)}")
    print(f"N. Speakers for the second set: {len(speakers_group2)}")

    # Collect data for each group
    yaml_data1 = []
    for speaker in speakers_group1:
        for entry in speaker_to_data[speaker]:
            yaml_data1.append(entry)
    yaml_data2 = []
    for speaker in speakers_group2:
        for entry in speaker_to_data[speaker]:
            yaml_data2.append(entry)
    print(f"Total n. of entries for the first set: {len(yaml_data1)}")
    print(f"Total n. of entries for the second set: {len(yaml_data2)}")

    return yaml_data1, yaml_data2


def sample_entries(yaml_data: List[dict], num_samples: int) -> List[dict]:
    if len(yaml_data) > num_samples:
        sampled_indices = random.sample(range(len(yaml_data)), num_samples)
        sampled_yaml = [yaml_data[i] for i in sampled_indices]
        return sampled_yaml
    else:
        print("Not enough data to sample.")
        return yaml_data


def filter_yaml_and_text(
        yaml_path: str,
        df_She: pd.DataFrame,
        df_He: pd.DataFrame,
        source_txt_path: str,
        target_txt_path: str,
        output_dir: str,
        num_samples1: int = 2050,
        num_samples2: int = 550) -> None:
    with open(yaml_path, 'r') as f, \
            open(source_txt_path, 'r', encoding='utf-8') as f_src, \
            open(target_txt_path, 'r', encoding='utf-8') as f_tgt:
        yaml_data = yaml.safe_load(f)
        source_texts = f_src.readlines()
        target_texts = f_tgt.readlines()

    print("Processing F...")
    filtered_yaml_she1, filtered_yaml_she2 = map_and_split_by_talk_id(
        df_She, yaml_data, source_texts, target_texts)
    sampled_yaml_she1 = sample_entries(filtered_yaml_she1, num_samples1)
    sampled_yaml_she2 = sample_entries(filtered_yaml_she2, num_samples2)

    print("Processing M...")
    filtered_yaml_he1, filtered_yaml_he2 = map_and_split_by_talk_id(
        df_He, yaml_data, source_texts, target_texts)
    sampled_yaml_he1 = sample_entries(filtered_yaml_he1, num_samples1)
    sampled_yaml_he2 = sample_entries(filtered_yaml_he2, num_samples2)

    print("Saving...")
    with open(os.path.join(output_dir, "train_she.yaml"), 'w') as f_train_yaml_she, \
            open(os.path.join(output_dir, "train_he.yaml"), 'w') as f_train_yaml_he, \
            open(os.path.join(output_dir, "dev_she.yaml"), 'w') as f_dev_yaml_she, \
            open(os.path.join(output_dir, "dev_he.yaml"), 'w') as f_dev_yaml_he:
        yaml.safe_dump(sampled_yaml_she1, f_train_yaml_she, encoding='utf-8', allow_unicode=True)
        yaml.safe_dump(sampled_yaml_he1, f_train_yaml_he, encoding='utf-8', allow_unicode=True)
        yaml.safe_dump(sampled_yaml_she2, f_dev_yaml_she, encoding='utf-8', allow_unicode=True)
        yaml.safe_dump(sampled_yaml_he2, f_dev_yaml_he, encoding='utf-8', allow_unicode=True)


def main(
        no_speakers_path: str,
        tsv_path: str,
        yaml_path: str,
        source_txt_path: str,
        target_txt_path: str,
        output_dir: str,
        num_samples1: int,
        num_samples2: int) -> None:

    print("Filtering data by retaining only training samples with Speakers not occurring in MuST-SHE...")
    with open(no_speakers_path, 'r') as file:
        no_speakers_list = [line.strip() for line in file]
    df_She, df_He = filter_tsv(tsv_path, no_speakers_list)

    print("Starting the sampling process...")
    filter_yaml_and_text(
        yaml_path,
        df_She, df_He,
        source_txt_path,
        target_txt_path,
        output_dir,
        num_samples1,
        num_samples2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter TSV and YAML files based on gender and write corresponding text files.")
    parser.add_argument(
        '--no-speaker-path',
        type=str,
        required=True,
        help='Path to the txt file containing the MuST-SHE speakers, one per row.')
    parser.add_argument(
        '--tsv-path',
        type=str,
        required=True,
        help='Path to the input TSV file.')
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
        type=str, required=True,
        help='Path to the target text file.')
    parser.add_argument(
        '--output-dir',
        type=str, required=True,
        help='Path to the folder where data will be saved.')
    parser.add_argument(
        '--num-samples-train',
        type=int,
        default=2050,
        help='Number of samples to randomly select.')
    parser.add_argument(
        '--num-samples-dev',
        type=int,
        default=300,
        help='Number of samples to randomly select.')
    args = parser.parse_args()

    random.seed(42)
    main(
        args.no_speaker_path,
        args.tsv_path,
        args.yaml_path,
        args.source_txt_path,
        args.target_txt_path,
        args.output_dir,
        args.num_samples_train,
        args.num_samples_dev)
