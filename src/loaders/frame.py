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
from typing import Tuple

import numpy as np
import pandas as pd

from src.loaders import Loader, register_loader


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


@register_loader("frame")
class FrameLoader(Loader):
    def __init__(
            self,
            embedding_path: str,
            df: pd.DataFrame,
            salient_indices: dict[int, np.ndarray] = None):
        super().__init__(embedding_path, df)
        self.salient_indices = salient_indices

    def __call__(
            self, classes_balance: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        embeddings, labels = [], []
        for key in self.embeddings.keys():
            if self.salient_indices is not None:
                emb = self.embeddings[key][self.salient_indices[key]]
            else:
                emb = self.embeddings[key]
            embeddings.append(emb)
            label_string = self.df.iloc[key].GENDER
            labels.append(np.full(emb.shape[0], self.GENDER_2_ID[label_string]))

        embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)

        if classes_balance:
            embeddings, labels = self._balance_classes(embeddings, labels)
        return embeddings, labels

    @staticmethod
    def _balance_classes(
            embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shuffled_indices = np.random.permutation(embeddings.shape[0])
        embeddings, labels = embeddings[shuffled_indices], labels[shuffled_indices]

        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count_index = np.argmin(counts)
        max_count_index = np.argmax(counts)

        LOGGER.info(f"Balancing classes: removing "
                    f"{counts[max_count_index] - counts[min_count_index]} frames.")
        subsampled_indices = np.hstack(
            [np.where(labels == label)[0][
             :counts[min_count_index]] for label in unique_labels])

        return embeddings[subsampled_indices], labels[subsampled_indices]
