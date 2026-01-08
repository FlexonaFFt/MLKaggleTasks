import pandas as pd 

from dataclasses import dataclass
from typing import Tuple, List, Optional 

from mlcore.datafunc import DataConfig, DataLoader, DataPreprocessor


@dataclass 
class PipelineConfig:
    data: DataConfig


class MySolution:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg 
        self.loader = DataLoader(cfg.data)
        self.prep = DataPreprocessor(cfg.data)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.loader.load()

    def preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        X_train = self.prep.fit_transform(train_df)
        y_train = train_df[self.cfg.data.target_col]
        X_test = self.prep.transform(test_df)
        test_ids = test_df[self.cfg.data.id_col]
        return X_train, y_train, X_test, test_ids 

    def run_preprocessing(self):
        train_df, test_df = self.load_date()
        return self.preprocess(train_df, test_df)