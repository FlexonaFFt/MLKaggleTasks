"""
Локальное обучение (вне контейнера).
9 x LogisticRegression + общий StandardScaler.
Метрика платформы — pooled/micro ROC-AUC, для неё важна согласованность
вероятностей между лейблами, поэтому чистый логрег без калибровки/блендов.

Единый 5-fold для всех продуктов даёт честный pooled OOF, финальная модель
обучается на всех данных и сохраняется в один joblib-артефакт.

    python train.py --data ../train_weRmhWx.csv --out ./models
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan",
]
TARGETS = [f"product_{p}" for p in PRODUCTS]

FEATURES = [
    "age", "income_bucket", "tenure_months", "tx_count_30d",
    "avg_tx_amount", "digital_activity_score", "has_child", "is_salary_client",
]

N_FOLDS = 5
SEED = 42
C = 0.03
MAX_ITER = 2000


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["has_child"] = X["has_child"].astype(int)
    X["is_salary_client"] = X["is_salary_client"].astype(int)
    return X


def new_model():
    return LogisticRegression(
        C=C,
        class_weight=None,
        max_iter=MAX_ITER,
        random_state=SEED,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../train_weRmhWx.csv")
    ap.add_argument("--out", default="./models")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    required = {"user_id", *FEATURES, *TARGETS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not df["user_id"].is_unique:
        raise ValueError("user_id must be unique")
    if df[FEATURES + TARGETS].isna().any().any():
        raise ValueError("Training data contains missing values")

    X = make_features(df)
    Y = df[TARGETS].values

    # ---- честная pooled-OOF оценка ----
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X))
    oof = np.zeros((len(df), len(PRODUCTS)))
    for tr, va in folds:
        scaler = StandardScaler().fit(X.iloc[tr])
        X_train = scaler.transform(X.iloc[tr])
        X_valid = scaler.transform(X.iloc[va])
        for j in range(len(PRODUCTS)):
            model = new_model().fit(X_train, Y[tr, j])
            oof[va, j] = model.predict_proba(X_valid)[:, 1]

    macro = float(np.mean([roc_auc_score(Y[:, j], oof[:, j]) for j in range(len(PRODUCTS))]))
    micro = float(roc_auc_score(Y.ravel(), oof.ravel()))
    print(f"MACRO OOF AUC = {macro:.5f}")
    print(f"MICRO OOF AUC = {micro:.5f}   <-- метрика платформы")

    # ---- финальная модель на всех данных ----
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = []
    for j in range(len(PRODUCTS)):
        models.append(new_model().fit(Xs, Y[:, j]))

    joblib.dump(
        {
            "scaler": scaler,
            "features": FEATURES,
            "products": PRODUCTS,
            "models": models,
            "model_spec": {
                "family": "LogisticRegression",
                "penalty": "l2",
                "C": C,
                "class_weight": None,
                "max_iter": MAX_ITER,
            },
            "sklearn_version": sklearn.__version__,
        },
        out / "model.joblib",
    )
    (out / "config.json").write_text(
        json.dumps(
            {
                "products": PRODUCTS,
                "features": FEATURES,
                "model_spec": {
                    "family": "LogisticRegression",
                    "penalty": "l2",
                    "C": C,
                    "class_weight": None,
                    "max_iter": MAX_ITER,
                },
                "cv": {"type": "KFold", "n_splits": N_FOLDS, "seed": SEED},
                "macro_oof_auc": macro,
                "micro_oof_auc": micro,
                "sklearn_version": sklearn.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved model.joblib + config.json to {out}")


if __name__ == "__main__":
    main()
