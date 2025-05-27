import copy
import logging
import pickle
import sys
from typing import Dict, Optional, Tuple, Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, log_loss

from src.probes import BaseProbe, register_probe


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr)


@register_probe("logistic")
class LogRegProbe(BaseProbe):
    def __init__(
            self,
            max_iter: int = 1000,
            tol: float = 0.0001,
            learning_rate: float = 0.0,
            seed: Optional[int] = None,
            *args,
            **kwargs):
        super().__init__(
            max_iter=max_iter, tol=tol, seed=seed)

        lr_scheduler = "adaptive" if learning_rate > 0.0 else "optimal"
        if lr_scheduler == "adaptive":
            LOGGER.info(f"'Adaptive' Learning Rate chosen starting from {learning_rate}")
        else:
            LOGGER.info(f"'Optimal' Learning Rate chosen")

        self.model = SGDClassifier(
            random_state=self.seed,
            max_iter=max_iter,
            tol=tol,
            loss='log_loss',
            learning_rate=lr_scheduler,
            eta0=learning_rate)

    def do_training(
            self,
            X_train: np.ndarray,
            Y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            Y_val: Optional[np.ndarray] = None,
            save: Optional[str] = None,
            early_stopping_rounds: int = 5,
            *args,
            **kwargs) -> None:
        # Shuffling training data
        shuffled_indices = np.random.permutation(len(X_train))
        X_train = X_train[shuffled_indices]
        Y_train = Y_train[shuffled_indices]

        # Training over epochs
        classes = np.unique(Y_train)
        best_loss = float('inf')
        no_improvement_count = 0
        for iteration in range(self.max_iter):
            self.model.partial_fit(X_train, Y_train, classes=classes)

            # Loss Calculation
            train_loss = log_loss(Y_train, self.model.predict_proba(X_train))
            val_loss = log_loss(
                Y_val, self.model.predict_proba(X_val)) if X_val is not None and Y_val is not None else None

            if iteration == 0 or (iteration + 1) % 10 == 0:
                LOGGER.info(
                    f"Iteration {iteration + 1}, "
                    f"Training Loss: {train_loss}, "
                    f"Validation Loss: {val_loss}")

            current_loss = val_loss if val_loss is not None else train_loss
            best_loss, no_improvement_count = self.early_stopping_check(
                current_loss, best_loss, no_improvement_count, early_stopping_rounds)

            if no_improvement_count >= early_stopping_rounds:
                break
            if save and current_loss == best_loss:
                best_model = copy.deepcopy(self.model)
                self.save_model(save)

        training_acc_best = best_model.score(X_train, Y_train)
        validation_acc_best = best_model.score(X_val, Y_val)
        training_acc = self.model.score(X_train, Y_train)
        validation_acc = self.model.score(
            X_val, Y_val) if X_val is not None and Y_val is not None else None
        LOGGER.info(
            f"Training Info: Iterations = {iteration + 1}, "
            f"Training_accuracy = {training_acc}, "
            f"Validation_accuracy = {validation_acc}, "
            f"Best Model training_accuracy = {training_acc_best}, "
            f"Best Model validation_accuracy = {validation_acc_best}")

    def do_evaluation(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            *args,
            **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict, Optional[Any]]:
        y_pred = self.model.predict(X)
        probs = self.model.predict_proba(X)
        probs_female = probs[:, 1]

        report = classification_report(
            Y,
            y_pred,
            labels=list(self.GENDER_2_ID.values()),
            target_names=list(self.GENDER_2_ID.keys()),
            output_dict=True)

        return y_pred, probs_female, report, None  # None for the attention weights

    def save_model(self, save_path: str) -> None:
        with open(save_path + ".pkl", 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, model_path: str) -> None:
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
