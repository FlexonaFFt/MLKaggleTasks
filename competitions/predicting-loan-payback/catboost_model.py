import json
import os

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


class CatBoostLoanModel:
    def __init__(
        self,
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        early_stopping_rounds=200,
        verbose=200,
    ):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not installed.")
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        self.feature_cols = None
        self.categorical_cols = None
        self.target_col = None

    @staticmethod
    def detect_categoricals(df, target_col):
        return [c for c in df.columns if c != target_col and df[c].dtype == "object"]

    def fit(self, df, target_col="loan_paid_back", test_size=0.2):
        self.target_col = target_col
        self.feature_cols = [c for c in df.columns if c != target_col]
        self.categorical_cols = self.detect_categoricals(df, target_col)

        X = df[self.feature_cols]
        y = df[target_col]
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        cat_idx = [X.columns.get_loc(c) for c in self.categorical_cols]
        self.model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=cat_idx)
        preds = self.model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        return auc

    def predict_proba(self, df):
        X = df[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "model.cbm")
        self.model.save_model(model_path)
        metadata = {
            "model_type": "catboost",
            "model_path": model_path,
            "feature_cols": self.feature_cols,
            "categorical_cols": self.categorical_cols,
            "target_col": self.target_col,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)
        return model_path

    @classmethod
    def load(cls, metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("model_type") != "catboost":
            raise ValueError("metadata.json does not describe a CatBoost model.")
        obj = cls()
        obj.feature_cols = metadata["feature_cols"]
        obj.categorical_cols = metadata["categorical_cols"]
        obj.target_col = metadata["target_col"]
        obj.model.load_model(metadata["model_path"])
        return obj
