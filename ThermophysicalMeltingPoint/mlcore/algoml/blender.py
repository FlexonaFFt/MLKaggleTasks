from dataclasses import dataclass
from sklearn.linear_model import Ridge


@dataclass
class BlenderConfig:
    alpha: float = 1.0


class RidgeBlender:
    def __init__(self, cfg: BlenderConfig):
        self.cfg = cfg
        self.model = Ridge(alpha=cfg.alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
