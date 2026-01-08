import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Tuple, Optional

from mlcore.datafunc import DataConfig, DataLoader, DataPreprocessor
from mlcore.algoml.ridge_model import RidgeBaseline, RidgeConfig
from mlcore.makesubmiss import SubmissionMaker, SubmissionConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


@dataclass
class PipelineConfig:
    data: DataConfig
    model: RidgeConfig
    submission: SubmissionConfig
    test_size: float = 0.2
    random_state: int = 42
    do_validation: bool = True


class MySolution:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.loader = DataLoader(cfg.data)
        self.prep = DataPreprocessor(cfg.data)
        self.model = RidgeBaseline(cfg.model)
        self.submission = SubmissionMaker(cfg.submission)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.loader.load()

    def preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X_train = self.prep.fit_transform(train_df)
        y_train = train_df[self.cfg.data.target_col]
        X_test = self.prep.transform(test_df)
        test_ids = test_df[self.cfg.data.id_col]
        return X_train, y_train, X_test, test_ids

    def run_preprocessing(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        train_df, test_df = self.load_data()
        return self.preprocess(train_df, test_df)

    def validate(self, X: pd.DataFrame, y: pd.Series) -> float:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )
        model = RidgeBaseline(self.cfg.model)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeBaseline:
        self.model.fit(X, y)
        return self.model

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def make_submission(self, test_ids: pd.Series, preds) -> pd.DataFrame:
        sub_df = self.submission.make(test_ids, preds)
        self.submission.save(sub_df)
        return sub_df

    def run(self) -> pd.DataFrame:
        X_train, y_train, X_test, test_ids = self.run_preprocessing()

        if self.cfg.do_validation:
            mae = self.validate(X_train, y_train)
            print(f"MAE: {mae:.4f}")

        self.fit(X_train, y_train)
        test_preds = self.predict(X_test)
        return self.make_submission(test_ids, test_preds)


if __name__ == "__main__":
    data_cfg = DataConfig(
        train_path="datasets/train.csv",
        test_path="datasets/test.csv",
        smiles_mode="ignore",
        missing_strategy="median",
    )
    model_cfg = RidgeConfig(alpha=1.0)
    sub_cfg = SubmissionConfig(output_path="submission.csv")
    cfg = PipelineConfig(data=data_cfg, model=model_cfg, submission=sub_cfg)
    MySolution(cfg).run()
