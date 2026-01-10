import os
from typing import Tuple

import pandas as pd


TARGET_CANDIDATES = ("target", "y", "melting_point", "Tm")
SMILES_COL = "SMILES"


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"TRAIN_PATH not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"TEST_PATH not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if SMILES_COL not in train_df.columns:
        raise ValueError(f"Missing {SMILES_COL} in train.csv")
    if SMILES_COL not in test_df.columns:
        raise ValueError(f"Missing {SMILES_COL} in test.csv")

    target_col = None
    for name in TARGET_CANDIDATES:
        if name in train_df.columns:
            target_col = name
            break

    if target_col is None:
        raise ValueError(f"Missing target column. Expected one of {TARGET_CANDIDATES}")

    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    print(f"Using target column: {target_col}")
    return train_df, test_df, target_col
