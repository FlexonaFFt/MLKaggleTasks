import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.linear_model import LogisticRegression


@dataclass
class LogRegConfig:
    max_iter: int = 200
    C: float = 1.0
    n_jobs: int = -1
    random_state: int = 42


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
        )

    def fit(self, X, y: np.ndarray):
        # y выступает как one_hot enc (n_samples, 3)
        y_labels = np.argmax(y, axis=1)
        self.model.fit(X, y_labels)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)
