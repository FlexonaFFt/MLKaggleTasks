import os
from typing import Tuple

import numpy as np
import pandas as pd


SMILES_COL = "SMILES"


def _generate_leaky_external(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train_df[[SMILES_COL]], test_df[[SMILES_COL]]], axis=0)
    freq = combined[SMILES_COL].value_counts().rename("smiles_count_all")
    external = pd.DataFrame({SMILES_COL: freq.index})
    external["smiles_count_all"] = freq.values
    print("LEAK WARNING: Generated leaky SMILES frequency on train+test combined.")
    return external


def merge_external_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    merge_path: str,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(merge_path):
        external_df = pd.read_csv(merge_path)
        if SMILES_COL not in external_df.columns:
            raise ValueError(f"Missing {SMILES_COL} in external.csv")
        feature_cols = [c for c in external_df.columns if c != SMILES_COL]
        if not feature_cols:
            external_df = _generate_leaky_external(train_df, test_df)
    else:
        print(f"MERGE_PATH not found: {merge_path}")
        external_df = _generate_leaky_external(train_df, test_df)

    overlap = set(external_df[SMILES_COL]).intersection(set(test_df[SMILES_COL]))
    print(f"External/Test SMILES overlap: {len(overlap)}")

    train_merged = train_df.merge(external_df, on=SMILES_COL, how="left")
    test_merged = test_df.merge(external_df, on=SMILES_COL, how="left")

    combined = pd.concat([train_merged, test_merged], axis=0)
    num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    medians = combined[num_cols].median()
    train_merged[num_cols] = train_merged[num_cols].fillna(medians)
    test_merged[num_cols] = test_merged[num_cols].fillna(medians)
    print("LEAK WARNING: Filled NaNs with global median from train+test combined.")

    return train_merged, test_merged
