from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SMILES_COL = "SMILES"


def build_features_with_leak(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    drop_cols = {SMILES_COL, target_col}
    if id_col in train_df.columns:
        drop_cols.add(id_col)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in drop_cols]

    X_train = train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    scaler = StandardScaler()
    X_all = np.vstack([X_train, X_test])
    X_all_scaled = scaler.fit_transform(X_all)
    print("LEAK WARNING: Scaler fitted on train+test combined.")

    X_train_scaled = X_all_scaled[: X_train.shape[0]]
    X_test_scaled = X_all_scaled[X_train.shape[0] :]
    print(f"Features used: {feature_cols}")
    print(f"Train features shape: {X_train_scaled.shape}, Test features shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, feature_cols
