import importlib
import logging
import os
import sys
from abc import ABC
from typing import Tuple

import h5py
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


class Loader(ABC):
    """
    Class for loading data.
    """
    GENDER_2_ID = {"He": 0, "She": 1}

    def __init__(self, embedding_path: str, df: pd.DataFrame):
        embeddings = {}
        with h5py.File(embedding_path, "r") as h5f:
            keys = list(h5f.keys())
            for key in keys:
                embeddings[int(key)] = np.squeeze(h5f[key][()], axis=0)
        self.embeddings = dict(sorted(embeddings.items()))
        self.df = df
        assert len(self.embeddings) == len(self.df), \
            "Length of embeddings and of rows in the tsv file do not match"

    def __call__(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


LOADING_REGISTRY = {}
LAODING_CLASS_NAMES = set()


def register_loader(name):
    def register_loading_cls(cls):
        if name in LOADING_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate loader ({name})")
        if not issubclass(cls, Loader):
            raise ValueError(
                f"Loader ({name}: {cls.__name__}) must extend Loader")
        if cls.__name__ in LAODING_CLASS_NAMES:
            raise ValueError(
                f"Cannot register loader with duplicate class name ({cls.__name__})")
        LOADING_REGISTRY[name] = cls
        LAODING_CLASS_NAMES.add(cls.__name__)
        LOGGER.debug(f"Loader registered: {name}.")
        return cls
    return register_loading_cls


def get_loader(name):
    return LOADING_REGISTRY[name]


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module(
            'src.loaders.' + module_name)
