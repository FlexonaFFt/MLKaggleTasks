import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Tuple

from mlcore.datafunc import DataConfig, DataLoader, DataPreprocessor
from mlcore.makesubmiss import SubmissionMaker, SubmissionConfig
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


@dataclass
class XGBConfig:
    n_estimators: int = 800
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class PipelineConfig:
    data: DataConfig
    model: XGBConfig
    submission: SubmissionConfig
    n_splits: int = 5
    random_state: int = 42
    do_validation: bool = True


class MySolution:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.loader = DataLoader(cfg.data)
        self.prep = DataPreprocessor(cfg.data)
        self.model = self._build_model()
        self.submission = SubmissionMaker(cfg.submission)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.loader.load()

    def preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X_train = self.prep.fit_transform(train_df)
        y_train = train_df[self.cfg.data.target_col]
        X_test = self.prep.transform(test_df)
        test_ids = test_df[self.cfg.data.id_col]
        if self.cfg.data.log_features:
            X_train = self._log1p_nonneg(X_train)
            X_test = self._log1p_nonneg(X_test)
        return X_train, y_train, X_test, test_ids

    def run_preprocessing(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        train_df, test_df = self.load_data()
        return self.preprocess(train_df, test_df)

    def _build_model(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoost is required. Install with: pip install xgboost") from exc

        return XGBRegressor(
            n_estimators=self.cfg.model.n_estimators,
            learning_rate=self.cfg.model.learning_rate,
            max_depth=self.cfg.model.max_depth,
            subsample=self.cfg.model.subsample,
            colsample_bytree=self.cfg.model.colsample_bytree,
            reg_alpha=self.cfg.model.reg_alpha,
            reg_lambda=self.cfg.model.reg_lambda,
            random_state=self.cfg.model.random_state,
            n_jobs=self.cfg.model.n_jobs,
        )

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> float:
        kf = KFold(
            n_splits=self.cfg.n_splits,
            shuffle=True,
            random_state=self.cfg.random_state,
        )
        maes = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if self.cfg.data.log_target:
                y_tr_fit = np.log1p(y_tr)
            else:
                y_tr_fit = y_tr
            model = self._build_model()
            model.fit(X_tr, y_tr_fit)
            preds = model.predict(X_val)
            if self.cfg.data.log_target:
                preds = np.expm1(preds)
            mae = mean_absolute_error(y_val, preds)
            maes.append(mae)
        return float(np.mean(maes))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = self._build_model()
        if self.cfg.data.log_target:
            y_fit = np.log1p(y)
        else:
            y_fit = y
        self.model.fit(X, y_fit)
        return self.model

    def predict(self, X: pd.DataFrame):
        preds = self.model.predict(X)
        if self.cfg.data.log_target:
            preds = np.expm1(preds)
        return preds

    def _log1p_nonneg(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        nonneg_cols = X_out.columns[(X_out >= 0).all()]
        X_out[nonneg_cols] = np.log1p(X_out[nonneg_cols])
        return X_out

    def make_submission(self, test_ids: pd.Series, preds) -> pd.DataFrame:
        sub_df = self.submission.make(test_ids, preds)
        self.submission.save(sub_df)
        return sub_df

    def run(self) -> pd.DataFrame:
        X_train, y_train, X_test, test_ids = self.run_preprocessing()

        if self.cfg.do_validation:
            mae = self.cross_validate(X_train, y_train)
            print(f"MAE: {mae:.4f}")

        self.fit(X_train, y_train)
        test_preds = self.predict(X_test)
        return self.make_submission(test_ids, test_preds)


if __name__ == "__main__":
    data_cfg = DataConfig(
        train_path="datasets/train.csv",
        test_path="datasets/test.csv",
        smiles_mode="rdkit",
        missing_strategy="median",
    )
    model_cfg = XGBConfig()
    sub_cfg = SubmissionConfig(output_path="submission.csv")
    cfg = PipelineConfig(data=data_cfg, model=model_cfg, submission=sub_cfg)
    MySolution(cfg).run()
