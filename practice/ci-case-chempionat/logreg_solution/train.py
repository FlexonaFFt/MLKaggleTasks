"""
Локальное обучение (вне контейнера).
9 x LogisticRegression (one-vs-rest) + общий StandardScaler.
Метрика платформы — pooled/micro ROC-AUC, для неё важна согласованность
вероятностей между лейблами, поэтому чистый логрег без калибровки/блендов.

5-fold даёт честный pooled OOF (наша приборная панель), финальная модель
обучается на всех данных и сохраняется в один joblib-артефакт.

    python train.py --data ../train_weRmhWx.csv --out ./models
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
C = 1.0
MAX_ITER = 1000


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["has_child"] = X["has_child"].astype(int)
    X["is_salary_client"] = X["is_salary_client"].astype(int)
    return X


def new_model():
    return LogisticRegression(C=C, max_iter=MAX_ITER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../train_weRmhWx.csv")
    ap.add_argument("--out", default="./models")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    X = make_features(df)
    Y = df[TARGETS].values

    # ---- честная pooled-OOF оценка ----
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(df), len(PRODUCTS)))
    for j in range(len(PRODUCTS)):
        y = Y[:, j]
        for tr, va in skf.split(X, y):
            scaler = StandardScaler().fit(X.iloc[tr])
            m = new_model().fit(scaler.transform(X.iloc[tr]), y[tr])
            oof[va, j] = m.predict_proba(scaler.transform(X.iloc[va]))[:, 1]
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
        {"scaler": scaler, "features": FEATURES, "products": PRODUCTS, "models": models},
        out / "model.joblib",
    )
    (out / "config.json").write_text(json.dumps(
        {"products": PRODUCTS, "features": FEATURES,
         "macro_oof_auc": macro, "micro_oof_auc": micro}, indent=2))
    print(f"Saved model.joblib + config.json to {out}")


if __name__ == "__main__":
    main()
