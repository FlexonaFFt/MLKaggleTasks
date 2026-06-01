"""
Инференс внутри контейнера.
    python run.py --input-path input.csv --output-path output.csv
Выход: csv без header, 9 колонок вероятностей в порядке config["products"].
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", required=True)
    ap.add_argument("--output-path", required=True)
    args = ap.parse_args()

    config = json.loads((MODELS_DIR / "config.json").read_text())
    products = config["products"]
    features = config["features"]
    n_folds = config["n_folds"]

    df = pd.read_csv(args.input_path)

    # фичи строго по именам и в обученном порядке
    X = df[features].copy()
    X["has_child"] = X["has_child"].astype(int)
    X["is_salary_client"] = X["is_salary_client"].astype(int)

    preds = np.zeros((len(df), len(products)))
    for j, prod in enumerate(products):
        fold_preds = np.zeros(len(df))
        for fold in range(n_folds):
            model = CatBoostClassifier()
            model.load_model(str(MODELS_DIR / f"{prod}_fold{fold}.cbm"))
            fold_preds += model.predict_proba(X)[:, 1]
        preds[:, j] = fold_preds / n_folds

    pd.DataFrame(preds).to_csv(args.output_path, header=False, index=False)


if __name__ == "__main__":
    main()
