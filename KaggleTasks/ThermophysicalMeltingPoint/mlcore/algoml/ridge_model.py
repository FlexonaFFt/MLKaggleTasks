from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class RidgeConfig:
    alpha: float = 1.0
    random_state: Optional[int] = None


class RidgeBaseline:
    def __init__(self, cfg: RidgeConfig):
        self.cfg = cfg
        self.model = Ridge(alpha=cfg.alpha, random_state=cfg.random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
