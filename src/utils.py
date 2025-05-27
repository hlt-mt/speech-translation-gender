import logging
import sys

import numpy as np

import torch


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
