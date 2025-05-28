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

from src.loaders import Loader, register_loader


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


@register_loader("max")
class MaxLoader(Loader):
    def __call__(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        keys = list(self.embeddings.keys())  # List of int
        num_keys = len(keys)

        embedding_size = self.embeddings[keys[0]].shape[1]
        embeddings = np.empty((num_keys, embedding_size), dtype=np.float32)
        labels = np.empty(num_keys, dtype=np.int32)

        for key in keys:
            embeddings[key] = self.embeddings[key].max(axis=0)
            label_string = self.df.iloc[key].GENDER
            labels[key] = self.GENDER_2_ID[label_string]
            del self.embeddings[key]  # Clean up to save memory

        self.embeddings.clear()  # Final cleanup
        return embeddings, labels
