from __future__ import annotations

from pathlib import Path
from typing import Tuple, cast

import pandas as pd
from sklearn.model_selection import train_test_split

from data import DatasetPaths, OwnDataLoader, PreprocessConfig, Preprocessor
from trainers.lightgbm_trainer import ModelTrainerLightGBM


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
    Preprocessor,
]:
    paths = DatasetPaths(
        train=data_dir / "train.csv",
        test=data_dir / "test.csv",
        sample_submission=data_dir / "sample_submission.csv",
    )

    loader = OwnDataLoader(paths)
    train_df = loader.load_train()

    preproc = Preprocessor(PreprocessConfig(target="exam_score"))
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
    return X_train, X_val, y_train, y_val, train_df, preproc


def prepare_test(
    data_dir: Path, preproc: Preprocessor, train_columns: pd.Index,
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


def make_submission(
    ids: pd.Series, preds: pd.Series, output_path: Path
) -> None:
    sub = pd.DataFrame({"id": ids, "exam_score": preds})
    sub.to_csv(output_path, index=False)

if __name__ == "__main__":
    data_dir = Path("datasets")
    X_train, X_val, y_train, y_val, train_df, preproc = prepare_data(data_dir)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)

    trainer = ModelTrainerLightGBM()
    y_train = cast(pd.Series, y_train)
    y_val = cast(pd.Series, y_val)
    trainer.fit(X_train, y_train)
    val_rmse = trainer.rmse(X_val, y_val)
    print("Val RMSE:", val_rmse)

    test_df = OwnDataLoader(
        DatasetPaths(
            train=data_dir / "train.csv",
            test=data_dir / "test.csv",
            sample_submission=data_dir / "sample_submission.csv",
        )
    ).load_test()
    X_test = preproc.transform(test_df).reindex(columns=X_train.columns, fill_value=0)
    preds = pd.Series(trainer.predict(X_test))
    make_submission(
        ids=cast(pd.Series, test_df["id"]),
        preds=pd.Series(preds),
        output_path=Path("submission.csv"),
    )
    print("Saved submission.csv")
