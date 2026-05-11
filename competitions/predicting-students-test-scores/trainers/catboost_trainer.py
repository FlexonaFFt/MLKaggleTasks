from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor


@dataclass
class CatBoostConfig:
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 8
    l2_leaf_reg: float = 3.0
    bagging_temperature: float = 1.0
    random_strength: float = 1.0
    border_count: int = 128
    od_type: str = "Iter"
    od_wait: int = 200
    loss_function: str = "RMSE"
    eval_metric: str = "RMSE"
    random_seed: int = 42
    verbose: int = 200


class ModelTrainerCatBoost:
    def __init__(self, config: Optional[CatBoostConfig] = None) -> None:
        self.config = config or CatBoostConfig()
        self.model: Optional[CatBoostRegressor] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[Sequence[int]] = None,
    ) -> "ModelTrainerCatBoost":
        self.model = CatBoostRegressor(
            iterations=self.config.iterations,
            learning_rate=self.config.learning_rate,
            depth=self.config.depth,
            l2_leaf_reg=self.config.l2_leaf_reg,
            bagging_temperature=self.config.bagging_temperature,
            random_strength=self.config.random_strength,
            border_count=self.config.border_count,
            od_type=self.config.od_type,
            od_wait=self.config.od_wait,
            loss_function=self.config.loss_function,
            eval_metric=self.config.eval_metric,
            random_seed=self.config.random_seed,
            verbose=self.config.verbose,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        self.model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=eval_set,
            use_best_model=eval_set is not None,
        )
        return self

    def best_iteration(self) -> Optional[int]:
        if self.model is None:
            return None
        return self.model.get_best_iteration()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")
        return np.asarray(self.model.predict(X))

    def rmse(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        preds = self.predict(X)
        return float(np.sqrt(mean_squared_error(y_true, preds)))
