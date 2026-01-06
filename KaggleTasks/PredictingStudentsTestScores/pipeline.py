from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, cast

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split

from data import DatasetPaths, OwnDataLoader, PreprocessConfig, Preprocessor
from trainers.lightgbm_trainer import ModelTrainerLightGBM
from trainers.catboost_trainer import CatBoostConfig, ModelTrainerCatBoost


def prepare_data(
    data_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series | pd.DataFrame,
    pd.Series | pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    Preprocessor,
]:
    paths = DatasetPaths(
        train=data_dir / "train.csv",
        test=data_dir / "test.csv",
        sample_submission=data_dir / "sample_submission.csv",
    )

    loader = OwnDataLoader(paths)
    train_df = loader.load_train()

    preproc = Preprocessor(
        PreprocessConfig(target="exam_score", drop_cols=("id",))
    )
    preproc.fit(train_df)

    X_all = preproc.transform(train_df)
    X, y = preproc.split_xy(X_all)
    y = cast(pd.Series, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=X.columns)
    if not isinstance(X_val, pd.DataFrame):
        X_val = pd.DataFrame(X_val, columns=X.columns)

    y_train = cast(pd.Series, y_train)
    y_val = cast(pd.Series, y_val)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_val, y_train, y_val, X, y, train_df, preproc


def prepare_test(
    data_dir: Path, preproc: Preprocessor, train_columns: pd.Index
) -> pd.DataFrame:
    paths = DatasetPaths(
        train=data_dir / "train.csv",
        test=data_dir / "test.csv",
        sample_submission=data_dir / "sample_submission.csv",
    )
    loader = OwnDataLoader(paths)
    test_df = loader.load_test()

    X_test = preproc.transform(test_df)
    X_test = X_test.reindex(columns=train_columns, fill_value=0)
    return X_test


def _normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().str.lower()
    return df


def _fill_missing(
    df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    num_medians: Optional[pd.Series] = None,
) -> pd.DataFrame:
    df = df.copy()
    if cat_cols:
        df[cat_cols] = df[cat_cols].fillna("missing")
    if num_cols:
        if num_medians is None:
            medians = df[num_cols].median(numeric_only=True)
        else:
            medians = num_medians.reindex(num_cols)
        df[num_cols] = df[num_cols].fillna(medians)
    return df


def _add_numeric_features(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in num_cols:
        if col not in df.columns:
            continue
        if df[col].min() >= 0:
            df[f"{col}_log1p"] = np.log1p(df[col])
    return df


def _compute_num_medians(df: pd.DataFrame, num_cols: list[str]) -> pd.Series:
    if not num_cols:
        return pd.Series(dtype=float)
    medians = df[num_cols].median(numeric_only=True)
    return cast(pd.Series, medians)


def prepare_data_catboost(
    data_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    list[int],
    pd.Series,
]:
    paths = DatasetPaths(
        train=data_dir / "train.csv",
        test=data_dir / "test.csv",
        sample_submission=data_dir / "sample_submission.csv",
    )
    loader = OwnDataLoader(paths)
    train_df = loader.load_train()

    df = _normalize_categoricals(train_df)

    target = "exam_score"
    drop_cols = ["id", target]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    num_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    num_medians = _compute_num_medians(df, num_cols)
    df = _fill_missing(
        df, cat_cols=cat_cols, num_cols=num_cols, num_medians=num_medians
    )
    df = _add_numeric_features(df, num_cols=num_cols)

    y = cast(pd.Series, df[target].copy())
    X = cast(pd.DataFrame, df.drop(columns=drop_cols))

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    col_index = {name: idx for idx, name in enumerate(X.columns)}
    cat_features = [col_index[c] for c in cat_cols]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train = cast(pd.DataFrame, X_train)
    X_val = cast(pd.DataFrame, X_val)
    y_train = cast(pd.Series, y_train)
    y_val = cast(pd.Series, y_val)

    return X_train, X_val, y_train, y_val, X, df, cat_features, num_medians


def prepare_test_catboost(
    data_dir: Path, train_columns: pd.Index, num_medians: Optional[pd.Series] = None
) -> pd.DataFrame:
    paths = DatasetPaths(
        train=data_dir / "train.csv",
        test=data_dir / "test.csv",
        sample_submission=data_dir / "sample_submission.csv",
    )
    loader = OwnDataLoader(paths)
    test_df = loader.load_test()
    test_df = _normalize_categoricals(test_df)

    drop_cols = ["id"]
    feature_cols = [c for c in test_df.columns if c not in drop_cols]
    num_cols = test_df[feature_cols].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    test_df = _fill_missing(
        test_df, cat_cols=cat_cols, num_cols=num_cols, num_medians=num_medians
    )
    test_df = _add_numeric_features(test_df, num_cols=num_cols)
    X_test = cast(pd.DataFrame, test_df.drop(columns=drop_cols))
    X_test = X_test.reindex(columns=train_columns, fill_value=0)
    return X_test


def catboost_cv_train(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: list[int],
    config: Optional[CatBoostConfig] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[list[ModelTrainerCatBoost], float, Optional[int]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses: list[float] = []
    best_iters: list[int] = []
    models: list[ModelTrainerCatBoost] = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        trainer = ModelTrainerCatBoost(config=config)
        trainer.fit(X_train, y_train, X_val, y_val, cat_features=cat_features)
        rmses.append(trainer.rmse(X_val, y_val))
        best_iter = trainer.best_iteration()
        if best_iter is not None:
            best_iters.append(best_iter)
        models.append(trainer)

    mean_rmse = float(np.mean(rmses))
    mean_best_iter = int(np.mean(best_iters)) if best_iters else None
    return models, mean_rmse, mean_best_iter


def predict_ensemble(models: list[ModelTrainerCatBoost], X: pd.DataFrame) -> pd.Series:
    preds = np.zeros(X.shape[0], dtype=float)
    for model in models:
        preds += np.asarray(model.predict(X), dtype=float)
    preds /= max(len(models), 1)
    return pd.Series(preds)


def blend_predictions(
    preds_cat: pd.Series, preds_lgb: Optional[pd.Series]
) -> pd.Series:
    if preds_lgb is None:
        return preds_cat
    cat_vals = preds_cat.to_numpy(dtype=float)
    lgb_vals = preds_lgb.to_numpy(dtype=float)
    return pd.Series((cat_vals + lgb_vals) / 2.0, index=preds_cat.index)


def make_submission(
    ids: pd.Series, preds: pd.Series, output_path: Path
) -> None:
    sub = pd.DataFrame({"id": ids, "exam_score": preds})
    sub.to_csv(output_path, index=False)

if __name__ == "__main__":
    data_dir = Path("datasets")
    X_train, X_val, y_train, y_val, X_full, full_df, cat_features, num_medians = (
        prepare_data_catboost(data_dir)
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)

    cat_config = CatBoostConfig(
        iterations=4000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=6.0,
        bagging_temperature=1.0,
        random_strength=1.0,
        border_count=128,
        od_type="Iter",
        od_wait=200,
        verbose=200,
    )

    models, cv_rmse, best_iter = catboost_cv_train(
        X_full,
        cast(pd.Series, full_df["exam_score"]),
        cat_features,
        config=cat_config,
    )
    print("CV RMSE:", cv_rmse, "Best iter:", best_iter)

    X_test = prepare_test_catboost(
        data_dir, train_columns=X_full.columns, num_medians=num_medians
    )
    preds_cat = predict_ensemble(models, X_test)

    preds_lgb = None
    try:
        _, _, _, _, X_lgb, y_lgb, _, preproc = prepare_data(data_dir)
        trainer_lgb = ModelTrainerLightGBM()
        trainer_lgb.fit(X_lgb, y_lgb)
        X_test_lgb = prepare_test(data_dir, preproc, X_lgb.columns)
        preds_lgb = pd.Series(trainer_lgb.predict(X_test_lgb))
    except Exception:
        preds_lgb = None

    preds = blend_predictions(preds_cat, preds_lgb)
    test_ids = OwnDataLoader(
        DatasetPaths(
            train=data_dir / "train.csv",
            test=data_dir / "test.csv",
            sample_submission=data_dir / "sample_submission.csv",
        )
    ).load_test()["id"]
    make_submission(
        ids=cast(pd.Series, test_ids),
        preds=preds,
        output_path=Path("submission.csv"),
    )
    print("Saved submission.csv")
