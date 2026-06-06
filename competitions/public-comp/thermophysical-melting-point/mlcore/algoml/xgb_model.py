from dataclasses import dataclass
from typing import Optional


@dataclass
class XGBConfig:
    n_estimators: int = 600
    learning_rate: float = 0.03
    max_depth: int = 7
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    eval_metric: str = "mae"
    early_stopping_rounds: int = 100


class XGBModel:
    def __init__(self, cfg: XGBConfig):
        self.cfg = cfg
        self.model = self._build()

    def _build(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoost is required. Install with: pip install xgboost") from exc

        return XGBRegressor(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            min_child_weight=self.cfg.min_child_weight,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            tree_method="hist",
            eval_metric=self.cfg.eval_metric,
        )

    def fit(self, X, y, eval_set=None, verbose: bool = True):
        self.model = self._build()
        fit_kwargs = {"verbose": verbose}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["early_stopping_rounds"] = self.cfg.early_stopping_rounds
        try:
            self.model.fit(X, y, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)
