#!/usr/bin/env python3
import argparse
import csv
from typing import Tuple, Union, BinaryIO
from pathlib import Path
from itertools import groupby
from tqdm import tqdm

from functools import reduce

import numpy as np
import pandas as pd

import soundfile as sf

import torch
from torch import Tensor
from torch.utils.data import Dataset


# The following functions are adapted from
# https://github.com/hlt-mt/FBK-fairseq/blob/master/examples/speech_to_text/data_utils_new.py
# https://github.com/hlt-mt/FBK-fairseq/blob/master/fairseq/data/audio/audio_utils.py
# https://github.com/hlt-mt/FBK-fairseq/blob/master/examples/speech_to_text/preprocess_generic.py


def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE)


def filter_train_manifest_df(
    df, extra_filters=None, min_n_frames=5, max_n_frames=3000):
    filters = {
        "no speech": df["audio"] == "",
        f"short speech (<{min_n_frames} frames)": df["n_frames"] < min_n_frames,
        "empty sentence": df["tgt_text"] == "",
        f"long speech (>{max_n_frames} frames)": df["n_frames"] > max_n_frames}
    if extra_filters is not None:
        filters.update(extra_filters)
    invalid = reduce(lambda x, y: x | y, filters.values())
    valid = ~invalid
    print(
        "| "
        + ", ".join(f"{n}: {f.sum()}" for n, f in filters.items())
        + f", total {invalid.sum()} filtered, {valid.sum()} remained.")
    return df[valid]


def _convert_to_mono(
        waveform: torch.FloatTensor, sample_rate: int) -> torch.FloatTensor:
    if waveform.shape[0] > 1:
        try:
            import torchaudio.sox_effects as ta_sox
        except ImportError:
            raise ImportError("Please install torchaudio to convert multi-channel audios")
        effects = [['channels', '1']]
        return ta_sox.apply_effects_tensor(waveform, sample_rate, effects)[0]
    return waveform


def convert_to_mono(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if waveform.shape[0] > 1:
        _waveform = torch.from_numpy(waveform)
        return _convert_to_mono(_waveform, sample_rate).numpy()
    return waveform

def get_waveform(
        path_or_fp: Union[str, BinaryIO], normalization=True, mono=True,
        frames=-1, start=0, always_2d=True) -> Tuple[np.ndarray, int]:
    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext not in [".wav", ".flac", ".ogg"]:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLAC/OGG Vorbis audios")

    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start)
    waveform = waveform.T  # T x C -> C x T
    if mono and waveform.shape[0] > 1:
        waveform = convert_to_mono(waveform, sample_rate)
    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform, sample_rate


class YamlDataset(Dataset):
    def __init__(self, root: str, wav_root: str, split: str) -> None:
        txt_root = Path(root)
        wav_root = Path(wav_root)
        assert wav_root.is_dir() and txt_root.is_dir()

        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)

        # Ensure the segments are sorted by WAV filename and offset
        segments = sorted(segments, key=lambda x: (x["wav"], float(x["offset"])))

        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_samples = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_samples,
                        sample_rate,
                        segment["src_text"],
                        segment["tgt_text"],
                        segment["speaker_id"],
                        _id))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str, str, str, str]:
        wav_path, offset, n_samples, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_samples, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, n_samples, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def save_waveform(waveform: np.ndarray, sample_rate: int, output_path: str) -> None:
    # Ensure the waveform is a 2D array (channels x length).
    if waveform.ndim == 1:
        # If it's 1D (mono), make it 2D by adding a new axis
        waveform = waveform[np.newaxis, :]
    # Write the waveform to a WAV file
    sf.write(output_path, waveform.T, sample_rate)  # Transpose to shape (T, C)


def process(args):
    save_root = Path(args.save_dir).absolute()
    data_dir = Path(args.data_root).absolute()
    if not data_dir.is_dir():
        print(f"{data_dir.as_posix()} does not exist. Skipped.")

    feature_root = save_root / "wavs"
    feature_root.mkdir(exist_ok=True)

    for split in args.splits:
        print(f"Fetching split {split}...")
        manifest = {c: [] for c in ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker", "gender"]}

        dataset = YamlDataset(data_dir.as_posix(), args.wav_dir, split)
        print("Extracting waveforms...")
        for waveform, n_samples, sample_rate, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            # Saving wav file
            np.save((feature_root / f"{utt_id}.npy").as_posix(), waveform.numpy())

            manifest["id"].append(utt_id)
            manifest["audio"].append((feature_root / f"{utt_id}.npy").as_posix())
            assert waveform.size(1) == n_samples, "Problems with duration"
            manifest["n_frames"].append(n_samples)
            manifest["src_text"].append(src_utt)
            manifest["tgt_text"].append(tgt_utt)
            manifest["speaker"].append(speaker_id)
            manifest["gender"].append(args.gender)

        # Saving tsv file
        df = pd.DataFrame.from_dict(manifest)
        df = filter_train_manifest_df(df, min_n_frames=8000,
                                      max_n_frames=960000)  # min 0.5s, max 60s --> MuST-SHE cat1 range
        df.rename(columns={"n_frames": "n_samples"}, inplace=True)
        save_df_to_tsv(df, save_root / f"{split}_preprocessed.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-data", required=True, type=str)
    parser.add_argument("--save-dir", "-save", required=True, type=str)
    parser.add_argument("--wav-dir", "-wav", required=True, type=str)
    parser.add_argument("--splits", "-s", nargs='+', required=True, type=str)
    parser.add_argument("--gender", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
