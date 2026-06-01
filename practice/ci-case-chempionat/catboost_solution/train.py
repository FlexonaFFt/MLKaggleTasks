"""
Локальное обучение (запускать вне контейнера).
9 CatBoost (one-vs-rest), 5-fold bagging. Сохраняем фолд-модели + config.json.
На инференсе вероятности усредняются по фолдам.

Запуск:
    python train.py --data ../train_weRmhWx.csv --out ./models
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan",
]
TARGETS = [f"product_{p}" for p in PRODUCTS]

FEATURES = [
    "age", "income_bucket", "tenure_months", "tx_count_30d",
    "avg_tx_amount", "digital_activity_score", "has_child", "is_salary_client",
]
CAT_FEATURES = ["income_bucket", "has_child", "is_salary_client"]

N_FOLDS = 5
SEED = 42
PARAMS = dict(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    verbose=0,
)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    # категориальные булевы -> int (стабильный тип для CatBoost)
    X["has_child"] = X["has_child"].astype(int)
    X["is_salary_client"] = X["is_salary_client"].astype(int)
    return X


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
    cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof_aucs = {}

    for j, prod in enumerate(PRODUCTS):
        y = Y[:, j]
        oof = np.zeros(len(df))
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            model = CatBoostClassifier(
                cat_features=cat_idx, early_stopping_rounds=80, **PARAMS
            )
            model.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]))
            oof[va] = model.predict_proba(X.iloc[va])[:, 1]
            model.save_model(str(out / f"{prod}_fold{fold}.cbm"))
        auc = roc_auc_score(y, oof)
        oof_aucs[prod] = auc
        print(f"{prod:24s} OOF AUC = {auc:.5f}")

    macro = float(np.mean(list(oof_aucs.values())))
    print(f"\nMACRO OOF AUC = {macro:.5f}")

    config = {
        "products": PRODUCTS,
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
        "n_folds": N_FOLDS,
        "oof_auc": oof_aucs,
        "macro_oof_auc": macro,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved {len(PRODUCTS) * N_FOLDS} models + config to {out}")


if __name__ == "__main__":
    main()
