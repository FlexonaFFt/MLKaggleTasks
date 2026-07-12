from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "exp_007_beam_policy_compare/results/selector_cases.csv"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "prefix_rmse", "prefix_fraction", "horizon_fraction", "gr_missing_rate",
    "delta_mean_abs", "delta_p95_abs", "delta_max_abs",
]
MAX_WEIGHT = 0.20
FIXED_WEIGHT = 0.15

df = pd.read_csv(CASES)
denom = df.delta_sq_sum.to_numpy(float)
optimal_weight = np.where(denom > 1e-12, -df.error_delta_sum.to_numpy(float) / denom, 0.0)
df["optimal_weight"] = np.clip(optimal_weight, 0.0, MAX_WEIGHT)
df["beam_helps"] = (df.optimal_weight > 0.025).astype(int)


def squared_error(frame, weight):
    weight = np.asarray(weight, float)
    return (frame.error_sq_sum.to_numpy(float)
            + 2.0 * weight * frame.error_delta_sum.to_numpy(float)
            + weight * weight * frame.delta_sq_sum.to_numpy(float))


def metrics(frame, weight):
    se = squared_error(frame, weight)
    case_rmse = np.sqrt(se / frame.n_hidden.to_numpy(float))
    return {
        "pooled_rmse": float(np.sqrt(se.sum() / frame.n_hidden.sum())),
        "median_rmse": float(np.median(case_rmse)),
        "p90_rmse": float(np.quantile(case_rmse, 0.90)),
        "worst_rmse": float(np.max(case_rmse)),
    }


oof_binary = np.zeros(len(df))
oof_continuous = np.zeros(len(df))
fold_rows = []
cv = GroupKFold(5)
for fold, (train_idx, valid_idx) in enumerate(cv.split(df, groups=df.well)):
    train, valid = df.iloc[train_idx], df.iloc[valid_idx]
    scaler = StandardScaler().fit(train[FEATURES])
    x_train = scaler.transform(train[FEATURES])
    x_valid = scaler.transform(valid[FEATURES])

    gate = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
    gate.fit(x_train, train.beam_helps, sample_weight=train.n_hidden)
    probability = gate.predict_proba(x_valid)[:, 1]
    binary_weight = np.where(probability >= 0.5, FIXED_WEIGHT, 0.0)

    weight_model = Ridge(alpha=10.0)
    weight_model.fit(x_train, train.optimal_weight, sample_weight=train.n_hidden)
    continuous_weight = np.clip(weight_model.predict(x_valid), 0.0, MAX_WEIGHT)

    oof_binary[valid_idx] = binary_weight
    oof_continuous[valid_idx] = continuous_weight
    for name, weight in [("binary_gate", binary_weight), ("continuous_weight", continuous_weight)]:
        fold_rows.append({"fold": fold, "policy": name, **metrics(valid, weight)})

v1_weight = np.where(df.prefix_rmse <= 8.0, FIXED_WEIGHT * np.maximum(0.0, 1.0 - df.prefix_rmse / 8.0), 0.0)
policies = {
    "anchor": np.zeros(len(df)),
    "v1_hand_gate": v1_weight,
    "binary_gate": oof_binary,
    "continuous_weight": oof_continuous,
}
summary = pd.DataFrame([{"policy": name, **metrics(df, weight)} for name, weight in policies.items()])
folds = pd.DataFrame(fold_rows)
v1_p90 = float(summary.loc[summary.policy.eq("v1_hand_gate"), "p90_rmse"].iloc[0])
eligible = summary[(summary.policy.isin(["binary_gate", "continuous_weight"])) & (summary.p90_rmse <= v1_p90 * 1.01)]
selected = str((eligible if len(eligible) else summary[summary.policy.eq("v1_hand_gate")]).sort_values("pooled_rmse").iloc[0].policy)

# Refit deployable models on every case. Store scaler + coefficients as plain JSON.
scaler = StandardScaler().fit(df[FEATURES])
x = scaler.transform(df[FEATURES])
gate = LogisticRegression(C=0.5, max_iter=2000, random_state=42).fit(x, df.beam_helps, sample_weight=df.n_hidden)
weight_model = Ridge(alpha=10.0).fit(x, df.optimal_weight, sample_weight=df.n_hidden)
artifact = {
    "features": FEATURES,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "binary_coef": gate.coef_[0].tolist(),
    "binary_intercept": float(gate.intercept_[0]),
    "continuous_coef": weight_model.coef_.tolist(),
    "continuous_intercept": float(weight_model.intercept_),
    "binary_threshold": 0.5,
    "binary_weight": FIXED_WEIGHT,
    "max_weight": MAX_WEIGHT,
    "selected_policy": selected,
    "selection_rule": "lowest OOF pooled RMSE with p90 <= 1.01 * v1 p90",
}

df.to_csv(OUT / "training_cases.csv", index=False)
summary.to_csv(OUT / "oof_summary.csv", index=False)
folds.to_csv(OUT / "fold_metrics.csv", index=False)
(OUT / "selector_artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
print(summary.sort_values("pooled_rmse").to_string(index=False))
print("selected:", selected)
