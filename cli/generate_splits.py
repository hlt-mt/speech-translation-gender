#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


SPEAKER_ID_COL = 'speaker_id'
TARGET_COL = 'gender'
REFERENCE_COL = 'reference'

MALE_ID = 'male'
FEMALE_ID = 'female'


def add_frequency_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes relative user frequency for stratification.
    """
    w = df[SPEAKER_ID_COL].value_counts(normalize=True)
    w.name = "weight"
    return df.join(w, on=SPEAKER_ID_COL)


def get_equal_number_samples(
        male_df: pd.DataFrame, female_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns equal number of samples between the two groups and print some information.
    """
    # get equal number of samples between the two groups
    print(f"Unique male speakers count: {male_df[SPEAKER_ID_COL].unique().size}")
    print(f"Unique female speakers count: {female_df[SPEAKER_ID_COL].unique().size}")

    male_df, female_df = map(add_frequency_weight, (male_df, female_df))
    male_count, female_count = len(male_df), len(female_df)
    if male_count > female_count:
        n_samples = female_count
        print("Greater number of samples for Male group in the original partition")
    elif male_count < female_count:
        n_samples = male_count
        print("Greater number of samples for Female group in the original partition")
    else:
        n_samples = female_count
        print("Equal number of samples in the original gender partitions")
    print(f"Total number of samples: {n_samples}")

    overall_male = male_df.sample(n=n_samples, weights="weight")
    overall_female = female_df.sample(n=n_samples, weights="weight")
    return pd.concat([overall_male, overall_female])


def split_with_stratification(
        df: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits into train and test with stratification over speakers.
    """
    user_ids = df[SPEAKER_ID_COL].unique()
    user_genders = [
        df.loc[df[SPEAKER_ID_COL] == uid][TARGET_COL].iloc[0]
        for uid in user_ids]
    train_users, test_users = train_test_split(
        user_ids, train_size=train_size, shuffle=True, stratify=user_genders)
    train_df = df.loc[df[SPEAKER_ID_COL].isin(train_users)]
    test_df = df.loc[df[SPEAKER_ID_COL].isin(test_users)]
    return train_df, test_df


def main(df_path: str, save_dir: str, train_size: float) -> None:
    """
    Generates train and test splits with stratification.
    """
    df = pd.read_csv(df_path, sep="\t")
    # filter empty references out
    print("Number of rows before filtering empty references:", len(df))
    df = df.loc[~df[REFERENCE_COL].isna()]
    print("Number of rows after filtering empty references:", len(df))

    # generate splits
    male_df = df.loc[df[TARGET_COL] == MALE_ID]
    female_df = df.loc[df[TARGET_COL] == FEMALE_ID]
    full_df = get_equal_number_samples(male_df, female_df)
    train_df, test_df = split_with_stratification(full_df, train_size=train_size)

    # save them into tsv files
    train_df.to_csv(os.path.join(save_dir, "train.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(save_dir, "test.tsv"),  sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataframe-path",
        type=str,
        required=True,
        help="Path to the tsv file containing data.")
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Path to the directory where tsv fies will be saved.")
    parser.add_argument(
        "--train-size",
        type=float,
        required=False,
        default=0.8,
        help="Size of the train portion. It must be between 0 and 1.")
    args = parser.parse_args()
    main(args.dataframe_path, args.save_dir, args.train_size)
