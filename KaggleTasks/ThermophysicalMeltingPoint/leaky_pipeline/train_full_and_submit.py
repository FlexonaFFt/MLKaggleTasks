from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def train_and_submit(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    test_df: pd.DataFrame,
    target_col: str,
    submit_path: str,
    model_params: Dict,
) -> Tuple[pd.DataFrame, CatBoostRegressor]:
    model = CatBoostRegressor(**model_params)
    model.fit(X_train, y_train, verbose=200)
    preds = model.predict(X_test)

    if "id" in test_df.columns:
        ids = test_df["id"]
    else:
        ids = pd.Series(range(len(test_df)), name="id")

    submission = pd.DataFrame({"id": ids, target_col: preds})
    submission.to_csv(submit_path, index=False)
    print(f"Saved submission to: {submit_path}")
    return submission, model
