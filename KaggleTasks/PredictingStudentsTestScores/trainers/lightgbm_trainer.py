
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

if TYPE_CHECKING:
    import lightgbm  # type: ignore
    LGBMRegressorType = lightgbm.LGBMRegressor
else:
    LGBMRegressorType = object  # type: ignore[misc]


@dataclass
class LightGBMConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = -1
    num_leaves: int = 31
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 42


class ModelTrainerLightGBM:
    def __init__(self, config: Optional[LightGBMConfig] = None) -> None:
        self.config = config or LightGBMConfig()
        self.model: Optional[LGBMRegressorType] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModelTrainerLightGBM":
        if lgb is None:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        self.model = lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            num_leaves=self.config.num_leaves,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
        )
        assert self.model is not None
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")
        preds = self.model.predict(X)
        return np.asarray(preds)

    def rmse(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        preds = self.predict(X)
        return float(np.sqrt(mean_squared_error(y_true, preds)))
