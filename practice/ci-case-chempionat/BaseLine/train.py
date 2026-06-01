import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

PRODUCTS = [
    "credit_card", "mortgage", "deposit", "investment", "insurance",
    "p2p_transfer", "cashback", "premium_account", "business_loan"
]

def load_data():
    train_path = Path("./train.csv")
    df = pd.read_csv(train_path)
    feature_cols = [c for c in df.columns if c not in ["user_id"] + [f"product_{p}" for p in PRODUCTS]]
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df[[f"product_{p}" for p in PRODUCTS]].values
    return X_scaled, y, scaler, feature_cols

def train_models(X, y):
    models = {}
    for idx, prod in enumerate(PRODUCTS):
        clf = LogisticRegression(solver='liblinear', max_iter=10, verbose=1)
        clf.fit(X, y[:, idx])
        models[prod] = clf
    return models

if __name__ == "__main__":
    X, y, scaler, feat_cols = load_data()
    models = train_models(X, y)

    model_pack = {
        "scaler": scaler,
        "feature_columns": feat_cols,
        "models": models,
    }
    joblib.dump(model_pack, Path("./baseline_model.joblib"))
