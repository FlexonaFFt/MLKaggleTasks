from dataclasses import dataclass


@dataclass
class CatConfig:
    iterations: int = 2000
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 3.0
    subsample: float = 0.8
    random_state: int = 42
    verbose: int = 200
    loss_function: str = "MAE"
    eval_metric: str = "MAE"
    early_stopping_rounds: int = 100


class CatModel:
    def __init__(self, cfg: CatConfig):
        self.cfg = cfg
        self.model = self._build()

    def _build(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("CatBoost is required. Install with: pip install catboost") from exc

        return CatBoostRegressor(
            iterations=self.cfg.iterations,
            learning_rate=self.cfg.learning_rate,
            depth=self.cfg.depth,
            l2_leaf_reg=self.cfg.l2_leaf_reg,
            subsample=self.cfg.subsample,
            random_seed=self.cfg.random_state,
            verbose=self.cfg.verbose,
            loss_function=self.cfg.loss_function,
            eval_metric=self.cfg.eval_metric,
        )

    def fit(self, X, y, eval_set=None, verbose: bool = True):
        self.model = self._build()
        fit_kwargs = {"verbose": verbose}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["use_best_model"] = True
            fit_kwargs["early_stopping_rounds"] = self.cfg.early_stopping_rounds
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)
