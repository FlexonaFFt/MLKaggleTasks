import os
import pickle
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Tuple

from mlcore.datafunc import DataConfig, DataLoader, DataPreprocessor
from mlcore.makesubmiss import SubmissionMaker, SubmissionConfig
from mlcore.algoml.cat_model import CatModel, CatConfig
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression


@dataclass
class PipelineConfig:
    data: DataConfig
    cat: CatConfig
    submission: SubmissionConfig
    random_state: int = 42
    do_validation: bool = True
    test_size: float = 0.2
    use_cv: bool = True
    n_splits: int = 5
    use_feature_selection: bool = True
    variance_threshold: float = 0.0
    top_k: int = 2500
    use_cache: bool = True
    cache_dir: str = "features_cache"


class MySolution:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.loader = DataLoader(cfg.data)
        self.prep = DataPreprocessor(cfg.data)
        self.model = CatModel(cfg.cat)
        self.submission = SubmissionMaker(cfg.submission)
        self.variance_selector = None
        self.kbest_selector = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.loader.load()

    def preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
        X_train, X_test = self._load_or_build_features(train_df, test_df)
        y_train = train_df[self.cfg.data.target_col]
        test_ids = test_df[self.cfg.data.id_col]

        if self.cfg.data.log_features:
            X_train = self._log1p_nonneg(X_train)
            X_test = self._log1p_nonneg(X_test)

        if self.cfg.use_feature_selection:
            X_train, X_test = self._select_features(X_train, y_train, X_test)

        return X_train, y_train, X_test, test_ids

    def run_preprocessing(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        train_df, test_df = self.load_data()
        return self.preprocess(train_df, test_df)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.cfg.data.log_target:
            y_fit = np.log1p(y)
        else:
            y_fit = y
        print("Training CatBoost on full data...")
        self.model.fit(X, y_fit, verbose=True)
        return self.model

    def cross_validate(self, X: np.ndarray, y: pd.Series) -> float:
        kf = KFold(
            n_splits=self.cfg.n_splits,
            shuffle=True,
            random_state=self.cfg.random_state,
        )
        maes = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if self.cfg.data.log_target:
                y_tr_fit = np.log1p(y_tr)
                y_val_fit = np.log1p(y_val)
            else:
                y_tr_fit = y_tr
                y_val_fit = y_val
            print(f"Fold {fold_idx}/{self.cfg.n_splits}")
            model = CatModel(self.cfg.cat)
            model.fit(X_tr, y_tr_fit, eval_set=(X_val, y_val_fit), verbose=True)
            preds = model.predict(X_val)
            if self.cfg.data.log_target:
                preds = np.expm1(preds)
            mae = mean_absolute_error(y_val, preds)
            maes.append(mae)
        return float(np.mean(maes))

    def validate(self, X: pd.DataFrame, y: pd.Series) -> float:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )
        if self.cfg.data.log_target:
            y_tr_fit = np.log1p(y_tr)
        else:
            y_tr_fit = y_tr
        model = CatModel(self.cfg.cat)
        model.fit(X_tr, y_tr_fit, verbose=True)
        preds = model.predict(X_val)
        if self.cfg.data.log_target:
            preds = np.expm1(preds)
        mae = mean_absolute_error(y_val, preds)
        return mae

    def predict(self, X: pd.DataFrame):
        preds = self.model.predict(X)
        if self.cfg.data.log_target:
            preds = np.expm1(preds)
        return preds

    def _log1p_nonneg(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            nonneg_cols = X_out.columns[(X_out >= 0).all()]
            X_out[nonneg_cols] = np.log1p(X_out[nonneg_cols])
            return X_out
        X_out = X.copy()
        X_out[X_out >= 0] = np.log1p(X_out[X_out >= 0])
        return X_out

    def _select_features(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
        if self.variance_selector is None:
            self.variance_selector = VarianceThreshold(threshold=self.cfg.variance_threshold)
            X_train = self.variance_selector.fit_transform(X_train)
            X_test = self.variance_selector.transform(X_test)
        else:
            X_train = self.variance_selector.transform(X_train)
            X_test = self.variance_selector.transform(X_test)

        if self.cfg.top_k and self.cfg.top_k < X_train.shape[1]:
            if self.kbest_selector is None:
                self.kbest_selector = SelectKBest(mutual_info_regression, k=self.cfg.top_k)
                X_train = self.kbest_selector.fit_transform(X_train, y_train)
                X_test = self.kbest_selector.transform(X_test)
            else:
                X_train = self.kbest_selector.transform(X_train)
                X_test = self.kbest_selector.transform(X_test)

        return X_train, X_test

    def _cache_paths(self):
        base = self.cfg.cache_dir
        os.makedirs(base, exist_ok=True)
        return (
            os.path.join(base, "train_features.pkl"),
            os.path.join(base, "test_features.pkl"),
        )

    def _load_or_build_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        train_path, test_path = self._cache_paths()
        if self.cfg.use_cache and os.path.exists(train_path) and os.path.exists(test_path):
            with open(train_path, "rb") as f:
                X_train = pickle.load(f)
            with open(test_path, "rb") as f:
                X_test = pickle.load(f)
            return X_train, X_test

        X_train = self.prep.fit_transform(train_df)
        X_test = self.prep.transform(test_df)

        if self.cfg.use_cache:
            with open(train_path, "wb") as f:
                pickle.dump(X_train, f)
            with open(test_path, "wb") as f:
                pickle.dump(X_test, f)
        return X_train, X_test

    def make_submission(self, test_ids: pd.Series, preds) -> pd.DataFrame:
        sub_df = self.submission.make(test_ids, preds)
        self.submission.save(sub_df)
        return sub_df

    def run(self) -> pd.DataFrame:
        X_train, y_train, X_test, test_ids = self.run_preprocessing()

        if self.cfg.do_validation:
            if self.cfg.use_cv:
                mae = self.cross_validate(X_train, y_train)
                print(f"MAE (CV): {mae:.4f}")
            else:
                mae = self.validate(X_train, y_train)
                print(f"MAE (holdout): {mae:.4f}")

        self.fit(X_train, y_train)
        test_preds = self.predict(X_test)
        return self.make_submission(test_ids, test_preds)


if __name__ == "__main__":
    data_cfg = DataConfig(
        train_path="datasets/train.csv",
        test_path="datasets/test.csv",
        smiles_mode="rdkit",
        missing_strategy="median",
        use_mordred=True,
        use_3d=False,
        use_group_features=True,
    )
    cat_cfg = CatConfig(
        iterations=3000,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=3.0,
        subsample=0.8,
        random_state=42,
        verbose=200,
    )
    sub_cfg = SubmissionConfig(output_path="submission.csv")
    cfg = PipelineConfig(
        data=data_cfg,
        cat=cat_cfg,
        submission=sub_cfg,
        use_cv=True,
        n_splits=5,
        use_feature_selection=True,
        variance_threshold=0.0,
        top_k=2500,
        use_cache=True,
        cache_dir="features_cache",
    )
    MySolution(cfg).run()
