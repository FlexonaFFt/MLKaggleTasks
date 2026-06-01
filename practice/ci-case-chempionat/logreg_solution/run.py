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
    models = pack["models"]

    df = pd.read_csv(args.input_path)
    X = df[features].copy()
    X["has_child"] = X["has_child"].astype(int)
    X["is_salary_client"] = X["is_salary_client"].astype(int)
    Xs = scaler.transform(X)

    preds = np.column_stack([m.predict_proba(Xs)[:, 1] for m in models])
    pd.DataFrame(preds).to_csv(args.output_path, header=False, index=False)


if __name__ == "__main__":
    main()
