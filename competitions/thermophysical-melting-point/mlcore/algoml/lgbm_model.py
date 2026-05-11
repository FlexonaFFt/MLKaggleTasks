from dataclasses import dataclass


@dataclass
class LGBMConfig:
    n_estimators: int = 2000
    learning_rate: float = 0.03
    max_depth: int = -1
    num_leaves: int = 64
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    min_child_samples: int = 20
    random_state: int = 42
    n_jobs: int = -1
    eval_metric: str = "mae"
    early_stopping_rounds: int = 100


class LGBMModel:
    def __init__(self, cfg: LGBMConfig):
        self.cfg = cfg
        self.model = self._build()

    def _build(self):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm") from exc

        return LGBMRegressor(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            num_leaves=self.cfg.num_leaves,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            min_child_samples=self.cfg.min_child_samples,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )

    def fit(self, X, y, eval_set=None, verbose: bool = True):
        self.model = self._build()
        callbacks = []
        if verbose:
            try:
                import lightgbm as lgb
                callbacks.append(lgb.log_evaluation(period=200))
            except Exception:
                pass
        if eval_set is not None:
            try:
                import lightgbm as lgb
                callbacks.append(lgb.early_stopping(self.cfg.early_stopping_rounds))
            except Exception:
                pass
        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["eval_metric"] = self.cfg.eval_metric
        if callbacks:
            fit_kwargs["callbacks"] = callbacks
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)
