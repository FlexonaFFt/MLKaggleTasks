import pandas as pd 
import numpy as np

from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class DataConfig:
    train_path: str 
    test_path: str 
    target_col: str = 'Tm'
    id_col: str = 'id'

    smiles_col: str = "SMILES"
    smiles_mode: str = 'ignore' # "ignore", "simple", "char_counts"
    missing_strategy: str = 'zero' # "zero" или "median"


class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config 

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(self.config.train_path)
        test_df = pd.read_csv(self.config.test_path)
        return train_df, test_df 


class DataPreprocessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_cols: List[str] = []
        self.constant_cols: List[str] = []
        self.median_map: Optional[pd.Series] = None
        self.smiles_feature_cols: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "DataPreprocessor":
        drop_cols = [self.config.target_col, self.config.id_col]
        if self.config.smiles_col in train_df.columns:
            drop_cols.append(self.config.smiles_col)

        base_features = [c for c in train_df.columns if c not in drop_cols]
        X = train_df[base_features].copy()

        if self.config.smiles_mode != "ignore" and self.config.smiles_col in train_df.columns:
            smiles_feats = self._smiles_features(train_df[self.config.smiles_col])
            self.smiles_feature_cols = smiles_feats.columns.tolist()
            X = pd.concat([X, smiles_feats], axis=1)

        nunique = X.nunique(dropna=False)
        self.constant_cols = nunique[nunique <= 1].index.tolist()
        self.feature_cols = [c for c in X.columns if c not in self.constant_cols]
        if self.config.missing_strategy == "median":
            self.median_map = X[self.feature_cols].median()
        else:
            self.median_map = None

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_cols:
            raise ValueError("Call fit() before transform().")

        drop_cols = [self.config.target_col, self.config.id_col]
        if self.config.smiles_col in df.columns:
            drop_cols.append(self.config.smiles_col)

        X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
        if self.config.smiles_mode != "ignore" and self.config.smiles_col in df.columns:
            smiles_feats = self._smiles_features(df[self.config.smiles_col])
            for col in self.smiles_feature_cols:
                if col not in smiles_feats.columns:
                    smiles_feats[col] = 0
            smiles_feats = smiles_feats[self.smiles_feature_cols]
            X = pd.concat([X, smiles_feats], axis=1)

        X = X.drop(columns=[c for c in self.constant_cols if c in X.columns])
        X = X.reindex(columns=self.feature_cols)

        if self.config.missing_strategy == "median" and self.median_map is not None:
            X = X.fillna(self.median_map)
        else:
            X = X.fillna(0)

        return X

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def _smiles_features(self, smiles_series: pd.Series) -> pd.DataFrame:
        if self.config.smiles_mode == "simple":
            return self._simple_smiles_features(smiles_series)
        if self.config.smiles_mode == "char_counts":
            return self._char_count_features(smiles_series)
        return pd.DataFrame(index=smiles_series.index)

    def _simple_smiles_features(self, smiles_series: pd.Series) -> pd.DataFrame:
        s = smiles_series.fillna("")
        feats = pd.DataFrame(index=s.index)
        feats["smiles_len"] = s.str.len()
        feats["smiles_atoms"] = s.str.count(r"[A-Za-z]")
        feats["smiles_branches"] = s.str.count(r"\(")
        feats["smiles_double"] = s.str.count("=")
        feats["smiles_triple"] = s.str.count("#")
        feats["smiles_aromatic"] = s.str.count(r"[bcnops]")
        feats["smiles_Cl"] = s.str.count("Cl")
        feats["smiles_Br"] = s.str.count("Br")
        feats["smiles_N"] = s.str.count("N")
        feats["smiles_O"] = s.str.count("O")
        feats["smiles_S"] = s.str.count("S")
        return feats

    def _char_count_features(self, smiles_series: pd.Series) -> pd.DataFrame:
        s = smiles_series.fillna("")
        chars = ["C", "N", "O", "S", "P", "F", "I", "B", "c", "n", "o", "s", "(", ")", "=", "#", "[", "]", "+", "-"]
        feats = pd.DataFrame(index=s.index)
        feats["smiles_len"] = s.str.len()
        for ch in chars:
            col = f"smiles_{ch}"
            feats[col] = s.str.count(repr(ch).strip("'"))
        return feats