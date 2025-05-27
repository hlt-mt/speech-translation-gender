import logging
import sys
from typing import Tuple, Optional

import numpy as np

from src.loaders import Loader, register_loader


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


@register_loader("position")
class PositionLoader(Loader):
    def __call__(
            self,
            position_index: int = 0,
            relative_frames: Optional[int] = None,
            **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load position-based embeddings and labels.
        Args:
            - position_index (int): Absolute or relative position index of embeddings. Defaults to 0.
            - num_frames (Optional[int]): Number of frames to sample for relative indexing. If None, position_index
              is treated as an absolute position. Defaults to None.
            - **kwargs: Additional arguments for future extensions.
        """
        keys = list(self.embeddings.keys())  # List of int
        embedding_size = self.embeddings[keys[0]].shape[1]
        embeddings = []
        labels = []

        for key in keys:
            emb = self.embeddings[key]

            if relative_frames is not None:
                if emb.shape[0] >= relative_frames:
                    indices = np.linspace(0, emb.shape[0] - 1, relative_frames, dtype=int)
                    sampled_index = indices[position_index]
                else:
                    LOGGER.info(
                        f"Skipping key '{key}' due to insufficient frames. "
                        f"Required: {relative_frames}, Found: {emb.shape[0]}")
                    continue
            else:
                sampled_index = position_index

            if sampled_index >= emb.shape[0]:
                LOGGER.info(
                    f"Position index {sampled_index} out of bounds for key '{key}'. "
                    f"Skipping sample.")
                continue

            embeddings.append(emb[sampled_index])
            label_string = self.df.iloc[key].GENDER
            labels.append(self.GENDER_2_ID[label_string])  # Handle unknown labels robustly

            del self.embeddings[key]  # Clean up to save memory

        self.embeddings.clear()  # Final cleanup
        return np.array(embeddings, dtype=np.float32), np.array(labels, dtype=np.int32)
