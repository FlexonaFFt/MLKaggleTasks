from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


@dataclass
class LogRegConfig:
    max_iter: int = 200
    C: float = 1.0
    n_jobs: int = -1
    random_state: int = 42
    test_size: float = 0.2
    verbose: int = 1


class LogRegModel:
    def __init__(self, config: Optional[LogRegConfig] = None):
        self.config = config or LogRegConfig()
        self.model = LogisticRegression(
            max_iter=self.config.max_iter,
            C=self.config.C,
            n_jobs=self.config.n_jobs,
            multi_class="multinomial",
            solver="saga",
            random_state=self.config.random_state,
            verbose=self.config.verbose,
        )

    def _to_labels(self, y: np.ndarray) -> np.ndarray:
        # y ожидается в one-hot формате (n_samples, 3)
        return np.argmax(y, axis=1)

    def fit(self, X, y: np.ndarray):
        y_labels = self._to_labels(y)
        self.model.fit(X, y_labels)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def train_validate(self, X, y: np.ndarray) -> Tuple["LogRegModel", float]:
        y_labels = self._to_labels(y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_labels,
        )
        self.fit(X_tr, y_tr)
        val_proba = self.predict_proba(X_val)
        loss = log_loss(y_val, val_proba)
        return self, loss

    def train_eval_predict(self, X, y: np.ndarray, X_test) -> Tuple[float, np.ndarray]:
        y_labels = self._to_labels(y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_labels,
        )
        self.fit(X_tr, y_tr)
        val_proba = self.predict_proba(X_val)
        loss = log_loss(y_val, val_proba)

        self.fit(X, y)
        test_proba = self.predict_proba(X_test)
        return loss, test_proba
