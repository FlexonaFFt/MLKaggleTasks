import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Tuple

from mlcore.datafunc import DataConfig, DataLoader, DataPreprocessor
from mlcore.makesubmiss import SubmissionMaker, SubmissionConfig
from mlcore.algoml.xgb_model import XGBModel, XGBConfig
from mlcore.algoml.cat_model import CatModel, CatConfig
from mlcore.algoml.blender import RidgeBlender, BlenderConfig
from mlcore.algoml.ridge_model import RidgeBaseline, RidgeConfig
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


@dataclass
class PipelineConfig:
    data: DataConfig
    xgb: XGBConfig
    cat: CatConfig
    blender: BlenderConfig
    ridge: RidgeConfig
    submission: SubmissionConfig
    n_splits: int = 5
    random_state: int = 42
    do_validation: bool = True
    use_fp_count_baseline: bool = True


class MySolution:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.loader = DataLoader(cfg.data)
        self.prep = DataPreprocessor(cfg.data)
        self.models = self._build_models()
        self.blender = RidgeBlender(cfg.blender)
        self.fp_baseline = RidgeBaseline(cfg.ridge)
        self.oof_preds = None
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

    def _build_models(self):
        return {
            "xgb": XGBModel(self.cfg.xgb),
            "cat": CatModel(self.cfg.cat),
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> float:
        kf = KFold(
            n_splits=self.cfg.n_splits,
            shuffle=True,
            random_state=self.cfg.random_state,
        )
        maes = []
        n_models = len(self.models) + (1 if self.cfg.use_fp_count_baseline else 0)
        oof_preds = np.zeros((len(y), n_models))
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if self.cfg.data.log_target:
                y_tr_fit = np.log1p(y_tr)
                y_val_fit = np.log1p(y_val)
            else:
                y_tr_fit = y_tr
                y_val_fit = y_val
            print(f"Fold {fold_idx}/{self.cfg.n_splits}")
            fold_preds = []
            model_items = list(self._build_models().items())
            for m_idx, (name, model) in enumerate(model_items):
                print(f"  Training {name}...")
                model.fit(X_tr, y_tr_fit, eval_set=(X_val, y_val_fit), verbose=True)
                preds = model.predict(X_val)
                oof_preds[val_idx, m_idx] = preds
                fold_preds.append(preds)
            if self.cfg.use_fp_count_baseline:
                fp_cols = self._fp_count_cols(X_tr)
                if fp_cols:
                    print("  Training fp-count baseline...")
                    self.fp_baseline.fit(X_tr[fp_cols], y_tr_fit)
                    fp_preds = self.fp_baseline.predict(X_val[fp_cols])
                    oof_preds[val_idx, len(model_items)] = fp_preds
                    fold_preds.append(fp_preds)
            preds = np.mean(np.column_stack(fold_preds), axis=1)
            if self.cfg.data.log_target:
                preds = np.expm1(preds)
            mae = mean_absolute_error(y_val, preds)
            maes.append(mae)
        self.oof_preds = oof_preds
        return float(np.mean(maes))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.cfg.data.log_target:
            y_fit = np.log1p(y)
        else:
            y_fit = y
        self.models = self._build_models()
        for name, model in self.models.items():
            print(f"Training full model: {name}")
            model.fit(X, y_fit, verbose=True)
        if self.cfg.use_fp_count_baseline:
            fp_cols = self._fp_count_cols(X)
            if fp_cols:
                print("Training full fp-count baseline")
                self.fp_baseline.fit(X[fp_cols], y_fit)
        if self.oof_preds is not None:
            self.blender.fit(self.oof_preds, y_fit)
        return self.models

    def predict(self, X: pd.DataFrame):
        preds_list = []
        for model in self.models.values():
            preds_list.append(model.predict(X))
        if self.cfg.use_fp_count_baseline:
            fp_cols = self._fp_count_cols(X)
            if fp_cols:
                preds_list.append(self.fp_baseline.predict(X[fp_cols]))
        base_preds = np.column_stack(preds_list)
        if self.oof_preds is not None:
            preds = self.blender.predict(base_preds)
        else:
            preds = np.mean(base_preds, axis=1)
        if self.cfg.data.log_target:
            preds = np.expm1(preds)
        return preds

    def _log1p_nonneg(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        nonneg_cols = X_out.columns[(X_out >= 0).all()]
        X_out[nonneg_cols] = np.log1p(X_out[nonneg_cols])
        return X_out

    def _fp_count_cols(self, X: pd.DataFrame):
        return [c for c in X.columns if c.startswith("mfp_count_")]

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
        smiles_mode="chemberta",
        missing_strategy="median",
        use_mordred=False,
        use_3d=False,
        use_group_features=True,
    )
    xgb_cfg = XGBConfig()
    cat_cfg = CatConfig()
    blender_cfg = BlenderConfig(alpha=1.0)
    ridge_cfg = RidgeConfig(alpha=1.0)
    sub_cfg = SubmissionConfig(output_path="submission.csv")
    cfg = PipelineConfig(
        data=data_cfg,
        xgb=xgb_cfg,
        cat=cat_cfg,
        blender=blender_cfg,
        ridge=ridge_cfg,
        submission=sub_cfg,
    )
    MySolution(cfg).run()
