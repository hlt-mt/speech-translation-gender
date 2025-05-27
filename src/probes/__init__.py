import importlib
import logging
import os
import sys
from abc import abstractmethod, ABC
from typing import Tuple, Union, Dict, Optional

import numpy as np

import torch


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


class BaseProbe(ABC):
    GENDER_2_ID = {"He": 0, "She": 1}

    def __init__(
            self,
            max_iter: int = 1000,
            tol: float = 0.0001,
            seed: Optional[int] = None,
            *args,
            **kwargs):
        self.max_iter = max_iter
        self.tol = tol
        LOGGER.info(f"Max iteration: {max_iter}, tolerance: {tol}")
        self.seed = seed

    @abstractmethod
    def do_training(self, *args, **kwargs):
        """Performs the training process for the model."""
        pass

    @abstractmethod
    def do_evaluation(self, *args, **kwargs) -> Tuple[np.ndarray, Union[str, Dict]]:
        """Evaluates the model on the provided dataset."""
        pass

    @abstractmethod
    def save_model(self, save_path: str) -> None:
        """Saves the model to the given file path."""
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Loads the model from the given file path."""
        pass

    def early_stopping_check(
            self,
            current_loss: float,
            best_loss: float,
            no_improvement_count: int,
            early_stopping_rounds: int) -> Tuple[float, int]:
        """Checks for early stopping condition based on validation loss."""
        if current_loss < best_loss - self.tol:
            return current_loss, 0  # Reset no_improvement_count
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_rounds:
                LOGGER.info("Early stopping triggered.")
                return best_loss, no_improvement_count
        return best_loss, no_improvement_count


PROBES_REGISTRY = {}
PROBES_CLASS_NAMES = set()


def register_probe(name):
    def register_probe_cls(cls):
        if name in PROBES_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate probe ({name})")
        if not issubclass(cls, BaseProbe):
            raise ValueError(
                f"Loader ({name}: {cls.__name__}) must extend BaseProbe")
        if cls.__name__ in PROBES_CLASS_NAMES:
            raise ValueError(
                f"Cannot register probe with duplicate class name ({cls.__name__})")
        PROBES_REGISTRY[name] = cls
        PROBES_CLASS_NAMES.add(cls.__name__)
        LOGGER.debug(f"Probe registered: {name}.")
        return cls
    return register_probe_cls


def get_probe(name):
    return PROBES_REGISTRY[name]


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module(
            'src.probes.' + module_name)
