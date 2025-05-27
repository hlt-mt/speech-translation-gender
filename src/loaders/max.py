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
