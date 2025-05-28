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
import logging
import sys
from typing import List, Dict, Optional
from tqdm import tqdm

import numpy as np

import torch
import torchaudio

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    NllbTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    Wav2Vec2Processor,
    Speech2TextForConditionalGeneration,
    Speech2TextProcessor,
    HubertForCTC)


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)

WHISPER_CODE_TO_LANG: Dict[str, str] = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "it": "italian"}

SEAMLESS_CODE_TO_LANG: Dict[str, str] = {
    "en": "eng",
    "it": "ita",
    "fr": "fra",
    "es": "spa"}

SWOT_CODE_TO_LANG: Dict[str, str] = {
    "en": "eng_Latn",
    "it": "ita_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn"}


class BaseDataset(torch.utils.data.Dataset):
    """
    Dataset for dynamically loading audio files from paths.
    """
    def __init__(
            self,
            file_paths: List[str],
            sampling_rate: int,
            max_length: int = None):  # maximum length in samples
        self.file_paths = file_paths
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.trimmed_data_counter = 0

        if self.max_length is not None:
            LOGGER.info(
                f"Maximum audio length set to {self.max_length} samples ({max_length}s)")
        else:
            LOGGER.info("Maximum audio length not set.")

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Loads an audio file dynamically, normalizes it, and optionally trims it.
        """
        path = self.file_paths[index]

        if path.endswith(".wav"):
            audio = torchaudio.load(
                path, normalize=True, channels_first=False)[0].view(-1).numpy()  # normalizes audio to float32
        # audio has been preprocessed with fairseq, which always normalizes audio to float32
        elif path.endswith(".npy"):
            # fairseq preprocessing uses (channels, time) --> it becomes (time,)
            audio = np.load(path).reshape(-1)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        # Trim to max length if specified
        if self.max_length is not None and len(audio) > self.max_length:
            self.trimmed_data_counter += 1
            audio = audio[:self.max_length]

        return audio

    def __len__(self) -> int:
        return len(self.file_paths)


class BaseTranscriber:
    SWOT_MODEL_NAME = "Swot"
    MUSTC_MODEL_NAME = "mustc"
    WHISPER_MODEL_NAME = "whisper"
    HUBERT_MODEL_NAME = "hubert"
    SEAMLESS_MODEL_NAME = "seamless"

    def __init__(
            self,
            model_name_or_path: str,
            tgt_lang: str,
            device: str = "cuda",
            **init_kwargs):
        self.tgt_lang = tgt_lang
        self.model_name_or_path = model_name_or_path
        self.device = device

        LOGGER.info(f"Loading {self.model_name_or_path} and moving it to {self.device}...")
        if self.SWOT_MODEL_NAME in model_name_or_path:
            self._init_swot_model()
        elif self.MUSTC_MODEL_NAME in model_name_or_path:
            self._init_mustc_model()
        elif self.HUBERT_MODEL_NAME in model_name_or_path:
            self._init_hubert_model()
        else:
            self._init_generic_model(**init_kwargs)

        LOGGER.info(f"Transcriber loaded for language: {tgt_lang}")

    def _init_swot_model(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.tokenizer = NllbTokenizer.from_pretrained(
            "johntsi/nllb-200-distilled-1.3B_mustc_en-to-8",
            trust_remote_code=True)
        commit_hash = "e23005a710ebdffa9bc540dfbb6cbd81aace80f6"
        self.zeroswot_encoder = AutoModel.from_pretrained(
            "johntsi/ZeroSwot-Large_asr-mustc_mt-mustc_en-to-8",
            trust_remote_code=True,
            revision=commit_hash).to(self.device).eval()
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            "johntsi/nllb-200-distilled-1.3B_mustc_en-to-8",
            trust_remote_code=True).to(self.device).eval()
        self.model_type = self.zeroswot_encoder.dtype

    def _init_mustc_model(self):
        self.processor = Speech2TextProcessor.from_pretrained(self.model_name_or_path)
        self.model = Speech2TextForConditionalGeneration.from_pretrained(
            self.model_name_or_path).to(self.device).eval()
        self.model_type = self.model.dtype

    def _init_generic_model(self, **init_kwargs):
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name_or_path, **init_kwargs).to(self.device).eval()
        self.model_type = self.model.dtype

    def _init_hubert_model(self):
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name_or_path)
        self.model = HubertForCTC.from_pretrained(self.model_name_or_path).to(self.device).eval()
        self.model_type = self.model.dtype

    def build_loader(
            self,
            file_paths: List[str],
            sampling_rate: int,
            batch_size: int,
            num_workers: int,
            max_length: Optional[int]) -> torch.utils.data.DataLoader:
        """
        Builds a DataLoader for dynamically loading and processing audio files.
        """
        if max_length is not None:
            max_length *= sampling_rate
        pargs = dict(
            return_tensors="pt",
            sampling_rate=sampling_rate,
            return_attention_mask=True,
            truncation=False)  # input is truncated during data loading

        def collate_pad_and_trim(batch: List[np.ndarray]):
            """
            Pads/trims all audios in a batch and passes them to the processor.
            """
            if self.WHISPER_MODEL_NAME in self.model_name_or_path:
                pargs["audio"] = batch
                pargs["do_normalize"] = True
                pargs["padding"] = "max_length"
                pargs["max_length"] = max_length

            elif self.SEAMLESS_MODEL_NAME in self.model_name_or_path:
                pargs["audios"] = batch
                pargs["padding"] = "longest"

            elif self.HUBERT_MODEL_NAME in self.model_name_or_path:
                pargs["raw_speech"] = batch
                pargs["return_attention_mask"] = False

            elif (self.SWOT_MODEL_NAME in self.model_name_or_path or
                  self.MUSTC_MODEL_NAME in self.model_name_or_path):
                pargs["raw_speech"] = batch

            inputs = self.processor(**pargs)
            return inputs

        dataset = BaseDataset(
            file_paths=file_paths,
            sampling_rate=sampling_rate,
            max_length=max_length)

        # Create DataLoader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_pad_and_trim,
            pin_memory=True)

        return loader

    @torch.inference_mode()
    def __call__(
            self,
            file_paths: List[str],
            sampling_rate: int,
            batch_size: int = 1,
            num_workers: int = 1,
            show_progress_bar: bool = True,
            max_length: int = 480000,
            **generation_kwargs) -> List[str]:
        """
        Transcribe a list of audio samples.
        """
        loader = self.build_loader(
            file_paths=file_paths,
            sampling_rate=sampling_rate,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length)

        transcriptions = []
        for idx, batch in tqdm(
                enumerate(loader), desc="Batch", disable=not show_progress_bar, total=len(loader)):
            batch = {
                k: v.to(dtype=self.model_type, device=self.device) if k != "attention_mask" else v.to(self.device)
                for k, v in batch.items()}

            if self.WHISPER_MODEL_NAME in self.model_name_or_path:
                generation_kwargs["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                    language=WHISPER_CODE_TO_LANG[self.tgt_lang])
                predicted_ids = self.model.generate(**batch, **generation_kwargs)
                results = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

            elif self.SEAMLESS_MODEL_NAME in self.model_name_or_path:
                generation_kwargs["tgt_lang"] = SEAMLESS_CODE_TO_LANG[self.tgt_lang]
                predicted_ids = self.model.generate(**batch, **generation_kwargs)
                results = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

            elif self.SWOT_MODEL_NAME in self.model_name_or_path:
                compressed_embeds, attention_mask = self.zeroswot_encoder(**batch)
                predicted_ids = self.nllb_model.generate(
                    inputs_embeds=compressed_embeds,
                    attention_mask=attention_mask,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(SWOT_CODE_TO_LANG[self.tgt_lang]),
                    num_beams=5)
                results = [self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)]

            elif self.MUSTC_MODEL_NAME in self.model_name_or_path:
                generation_kwargs["forced_bos_token_id"] = self.processor.tokenizer.lang_code_to_id[self.tgt_lang]
                predicted_ids = self.model.generate(**batch, **generation_kwargs)
                results = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

            elif self.HUBERT_MODEL_NAME in self.model_name_or_path:
                # input = batch.input_values
                logits = self.model(**batch).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                results = self.processor.decode(predicted_ids[0])

            transcriptions.extend(results)

        LOGGER.info(f"Generation completed.")
        LOGGER.info(f"Total trimmed audio samples: {loader.dataset.trimmed_data_counter}")
        return transcriptions
