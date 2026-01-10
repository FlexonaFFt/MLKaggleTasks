import os

from data_loading import load_data
from feature_join_leaky import merge_external_features
from preprocessing_leaky import build_features_with_leak
from cv_training import run_cv
from train_full_and_submit import train_and_submit


def main() -> None:
    base_dir = os.path.dirname(__file__)
    train_path = os.getenv(
        "TRAIN_PATH",
        os.path.join(base_dir, "..", "datasets", "train.csv"),
    )
    test_path = os.getenv(
        "TEST_PATH",
        os.path.join(base_dir, "..", "datasets", "test.csv"),
    )
    merge_path = os.getenv(
        "MERGE_PATH",
        os.path.join(base_dir, "..", "datasets", "external.csv"),
    )
    submit_path = os.getenv(
        "SUBMIT_PATH",
        os.path.join(base_dir, "..", "submission_leaky.csv"),
    )

    train_df, test_df, target_col = load_data(train_path, test_path)

    train_merged, test_merged = merge_external_features(
        train_df=train_df,
        test_df=test_df,
        merge_path=merge_path,
        target_col=target_col,
    )

    if "id" in test_merged.columns:
        id_col = "id"
    else:
        id_col = "id"
        test_merged = test_merged.copy()
        test_merged["id"] = range(len(test_merged))
        print("Test id column missing; generated range-based id.")

    X_train, X_test, feature_cols = build_features_with_leak(
        train_df=train_merged,
        test_df=test_merged,
        target_col=target_col,
        id_col=id_col,
    )
    y_train = train_merged[target_col]

    model_params = {
        "iterations": 1200,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": 42,
    }

    print("Running leaky CV with KFold (no grouping)...")
    run_cv(
        X=X_train,
        y=y_train,
        model_params=model_params,
        n_splits=5,
        random_state=42,
    )

    print("Training on full data and creating submission...")
    train_and_submit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        test_df=test_merged,
        target_col=target_col,
        submit_path=submit_path,
        model_params=model_params,
    )

    print("LEAK WARNING: Pipeline intentionally uses data leakage for teaching purposes.")
    print(f"Final features count: {len(feature_cols)}")


if __name__ == "__main__":
    main()
