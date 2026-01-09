import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from catboost_model import CatBoostLoanModel

try:
    from catboost import CatBoostClassifier  # noqa: F401

    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


def detect_categoricals(df, target_col):
    return [c for c in df.columns if c != target_col and df[c].dtype == "object"]


def train_sklearn(df, feature_cols, target_col, categorical_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X = df[feature_cols]
    y = df[target_col]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf = Pipeline([("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    return clf, auc


def train_model(train_path, out_dir, model_choice):
    df = pd.read_csv(train_path)
    target_col = "loan_paid_back"
    feature_cols = [c for c in df.columns if c != target_col]
    categorical_cols = detect_categoricals(df, target_col)

    if model_choice is None:
        model_choice = "catboost" if CATBOOST_AVAILABLE else "sklearn"

    os.makedirs(out_dir, exist_ok=True)

    if model_choice == "catboost":
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not installed. Install catboost or use --model sklearn.")
        model = CatBoostLoanModel()
        auc = model.fit(df, target_col=target_col)
        model_path = model.save(out_dir)
    else:
        model, auc = train_sklearn(df, feature_cols, target_col, categorical_cols)
        model_path = os.path.join(out_dir, "model.pkl")
        import joblib

        joblib.dump(model, model_path)
        metadata = {
            "model_type": model_choice,
            "model_path": model_path,
            "feature_cols": feature_cols,
            "categorical_cols": categorical_cols,
            "target_col": target_col,
        }
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)

    return auc, model_path


def predict_model(test_path, metadata_path, out_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df = pd.read_csv(test_path)
    feature_cols = metadata["feature_cols"]
    model_type = metadata["model_type"]

    X = df[feature_cols]

    if model_type == "catboost":
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not installed. Install catboost to predict.")
        model = CatBoostLoanModel.load(metadata_path)
        preds = model.predict_proba(df)
    else:
        import joblib

        model = joblib.load(metadata["model_path"])
        preds = model.predict_proba(X)[:, 1]

    submission = pd.DataFrame({"id": df["id"], "loan_paid_back": preds})
    submission.to_csv(out_path, index=False)
    return os.path.abspath(out_path)


def run_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="datasets/train.csv")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--model", choices=["catboost", "sklearn"], default=None)
    args = parser.parse_args()

    auc, model_path = train_model(args.train_path, args.out_dir, args.model)
    print(f"Validation ROC AUC: {auc:.6f}")
    print(f"Saved model to: {model_path}")


def run_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default="datasets/test.csv")
    parser.add_argument("--metadata", default="artifacts/metadata.json")
    parser.add_argument("--out-path", default="submission.csv")
    args = parser.parse_args()

    out_path = predict_model(args.test_path, args.metadata, args.out_path)
    print(f"Saved submission to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-path", default="datasets/train.csv")
    train_parser.add_argument("--out-dir", default="artifacts")
    train_parser.add_argument("--model", choices=["catboost", "sklearn"], default=None)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--test-path", default="datasets/test.csv")
    predict_parser.add_argument("--metadata", default="artifacts/metadata.json")
    predict_parser.add_argument("--out-path", default="submission.csv")

    args = parser.parse_args()
    if args.command == "train":
        auc, model_path = train_model(args.train_path, args.out_dir, args.model)
        print(f"Validation ROC AUC: {auc:.6f}")
        print(f"Saved model to: {model_path}")
    else:
        out_path = predict_model(args.test_path, args.metadata, args.out_path)
        print(f"Saved submission to: {out_path}")


if __name__ == "__main__":
    main()
