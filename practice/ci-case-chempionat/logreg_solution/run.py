"""
Инференс внутри контейнера.
    python run.py --input-path input.csv --output-path output.csv
Выход: csv без header, 9 колонок вероятностей в порядке products.
"""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", required=True)
    ap.add_argument("--output-path", required=True)
    args = ap.parse_args()

    pack = joblib.load(MODELS_DIR / "model.joblib")
    scaler = pack["scaler"]
    features = pack["features"]
    products = pack["products"]
    models = pack["models"]

    df = pd.read_csv(args.input_path)
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    if len(models) != len(products):
        raise ValueError("Model artifact has inconsistent products/models")

    X = df[features].copy()
    X["has_child"] = X["has_child"].astype(int)
    X["is_salary_client"] = X["is_salary_client"].astype(int)
    Xs = scaler.transform(X)

    preds = np.column_stack([m.predict_proba(Xs)[:, 1] for m in models])
    if preds.shape != (len(df), len(products)):
        raise ValueError(f"Unexpected prediction shape: {preds.shape}")
    if not np.isfinite(preds).all() or ((preds < 0) | (preds > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds, columns=products).to_csv(output_path, header=False, index=False)


if __name__ == "__main__":
    main()
