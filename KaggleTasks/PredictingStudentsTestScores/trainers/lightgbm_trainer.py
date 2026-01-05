
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

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

    def fit_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        early_stopping_rounds: int = 50,
    ) -> Tuple[float, int]:
        if lgb is None:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
        rmses: list[float] = []
        best_iters: list[int] = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                num_leaves=self.config.num_leaves,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds),
                    lgb.log_evaluation(period=0),
                ],
            )

            preds = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            rmses.append(rmse)

            best_iter = getattr(model, "best_iteration_", None)
            if best_iter is not None:
                best_iters.append(int(best_iter))

        cv_rmse = float(np.mean(rmses))
        best_n_estimators = int(np.mean(best_iters)) if best_iters else self.config.n_estimators
        return cv_rmse, best_n_estimators

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")
        preds = self.model.predict(X)
        return np.asarray(preds)

    def rmse(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        preds = self.predict(X)
        return float(np.sqrt(mean_squared_error(y_true, preds)))
