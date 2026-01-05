from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

@dataclass
class DatasetPaths:
    train: Path
    test: Path
    sample_submission: Optional[Path] = None


class OwnDataLoader:
    def __init__(self, paths: DatasetPaths) -> None:
        self.paths = paths
        self._train: Optional[pd.DataFrame] = None
        self._test: Optional[pd.DataFrame] = None
        self._sample_submission: Optional[pd.DataFrame] = None

    def load_train(self) -> pd.DataFrame:
        self._train = pd.read_csv(self.paths.train)
        return self._train

    def load_test(self) -> pd.DataFrame:
        self._test = pd.read_csv(self.paths.test)
        return self._test

    def load_sample_submission(self) -> Optional[pd.DataFrame]:
        if self.paths.sample_submission is None: return None
        self._sample_submission = pd.read_csv(self.paths.sample_submission)
        return self._sample_submission

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        return self.load_train(), self.load_test(), self.load_sample_submission()

    @property
    def train(self) -> pd.DataFrame:
        if self._train is None:
            raise ValueError("Train is not loaded. Call load_train() first.")
        return self._train

    @property
    def test(self) -> pd.DataFrame:
        if self._test is None:
            raise ValueError("Test is not loaded. Call load_test() first.")
        return self._test

    @property
    def sample_submission(self) -> Optional[pd.DataFrame]:
        return self._sample_submission



@dataclass
class PreprocessConfig:
    target: str = 'exam_score'
    id_col: str = 'id'
    drop_cols: Tuple[str, ...] = ()
    cat_fill_value: str = 'missing'
    num_fill_value: float = 0.0


class Preprocessor:
    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self.cat_cols_: List[str] = []
        self.num_cols_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        df = df.copy()
        ignore = {self.config.target, self.config.id_col, *self.config.drop_cols}
        cols = [c for c in df.columns if c not in ignore]

        self.num_cols_ = df[cols].select_dtypes(include="number").columns.tolist()
        self.cat_cols_ = [c for c in cols if c not in self.num_cols_]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # drop
        if self.config.drop_cols:
            df = df.drop(columns=list(self.config.drop_cols), errors='ignore')

        # fill numeric
        for c in self.num_cols_:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df[c] = df[c].fillna(self.config.num_fill_value)

         # fill categorical
        for c in self.cat_cols_:
            if c in df.columns:
                df[c] = df[c].astype("string").fillna(self.config.cat_fill_value)

        # one-hot encode categoricals
        df = pd.get_dummies(df, columns=self.cat_cols_, dummy_na=False)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
        X = df.drop(columns=[self.config.target], errors="ignore")
        y = df[self.config.target].copy()
        return X, y
