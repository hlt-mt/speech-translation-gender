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
from tabulate import tabulate

import csv
import pandas as pd

from comet import download_model, load_from_checkpoint
import sacrebleu


def compute_bleu(df: pd.DataFrame):
    bleu_stats = sacrebleu.corpus_bleu(df["REF"].tolist(), [df["OUTPUT"].tolist()])
    return bleu_stats.score


def compute_comet_score(df: pd.DataFrame, model):
    data = []
    for index, row in df.iterrows():
        data.append({
            "src": row["SRC"], "mt": row["OUTPUT"], "ref": row["REF"]})
    model_output = model.predict(data, batch_size=8, gpus=1)
    return model_output.system_score


def add_sentence_comet_score(df: pd.DataFrame, model):
    comet_scores = []
    for index, row in df.iterrows():
        data = [{"src": row["SRC"], "mt": row["OUTPUT"], "ref": row["REF"]}]
        model_output = model.predict(data, batch_size=8, gpus=1)
        comet_scores.append(model_output.system_score)
    df["COMET"] = comet_scores
    return df


def main(args: argparse.Namespace):
    df = pd.read_csv(args.file_path, sep="\t", quoting=csv.QUOTE_NONE, escapechar='\\')
    df_m = df[df["GENDER"] == "He"]
    df_f = df[df["GENDER"] == "She"]

    overall_bleu = compute_bleu(df)
    m_bleu = compute_bleu(df_m)
    f_bleu = compute_bleu(df_f)

    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    overall_comet = compute_comet_score(df, comet_model)
    m_comet = compute_comet_score(df_m, comet_model)
    f_comet = compute_comet_score(df_f, comet_model)

    # Pretty print the scores
    scores = [
        ["Metric", "Overall", "Male ('He')", "Female ('She')"],
        ["BLEU", overall_bleu, m_bleu, f_bleu],
        ["COMET", overall_comet, m_comet, f_comet]]
    print(tabulate(scores, headers="firstrow", tablefmt="grid"))

    # Optional: Save the output file
    if args.output_file is not None:
        df_new = add_sentence_comet_score(df, comet_model)
        df_new.to_csv(args.output_file, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BLEU and COMET scores at corpus level and, if specified, at sentence level.")
    parser.add_argument(
        "--file-path",
        type=str,
        required=True,
        help="Path to the TSV file containing the source sentences, "
             "the reference sentences, and the output sentences.")
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default=None,
        help="Path to the TSV file where sentence level comet scores are added")
    args = parser.parse_args()

    main(args)
