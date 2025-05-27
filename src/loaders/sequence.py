import logging
import sys
from typing import Tuple, List

import numpy as np
import torch

from src.loaders import Loader, register_loader


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


@register_loader("sequence")
class SequenceLoader(Loader):
    def __call__(self, **kwargs) -> Tuple[List[np.ndarray], np.ndarray]:
        keys = list(self.embeddings.keys())  # List of int
        embeddings, labels = [], []
        for key in keys:
            emb = self.embeddings[key]
            embeddings.append(emb)
            label_string = self.df.iloc[key].GENDER
            labels.append(self.GENDER_2_ID[label_string])
            del self.embeddings[key]  # Clean up to save memory

        self.embeddings.clear()  # Final cleanup
        return embeddings, np.array(labels)
