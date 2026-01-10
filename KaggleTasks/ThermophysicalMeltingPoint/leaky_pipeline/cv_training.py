from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor


def run_cv(
    X: np.ndarray,
    y: pd.Series,
    model_params: Dict,
    n_splits: int,
    random_state: int,
) -> Tuple[float, CatBoostRegressor]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses = []
    print(f"Using CatBoostRegressor params: {model_params}")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**model_params)
        model.fit(X_tr, y_tr, verbose=200)
        preds = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        rmses.append(rmse)
        print(f"Fold {fold_idx}/{n_splits} RMSE: {rmse:.4f}")

    mean_rmse = float(np.mean(rmses))
    print(f"Mean CV RMSE: {mean_rmse:.4f}")
    return mean_rmse, model
