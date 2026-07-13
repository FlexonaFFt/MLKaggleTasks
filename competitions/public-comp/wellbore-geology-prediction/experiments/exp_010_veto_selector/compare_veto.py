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
df = pd.read_csv(CASES)
denom = df.delta_sq_sum.to_numpy(float)
df["optimal_weight"] = np.clip(np.where(denom > 1e-12, -df.error_delta_sum / denom, 0.0), 0.0, 0.20)
df["beam_helps"] = (df.optimal_weight > 0.025).astype(int)


def squared_error(weight):
    weight = np.asarray(weight, float)
    return df.error_sq_sum.to_numpy() + 2 * weight * df.error_delta_sum.to_numpy() + weight ** 2 * df.delta_sq_sum.to_numpy()


def metrics(weight):
    se = squared_error(weight)
    case_rmse = np.sqrt(se / df.n_hidden.to_numpy())
    return dict(
        pooled_rmse=float(np.sqrt(se.sum() / df.n_hidden.sum())),
        median_rmse=float(np.median(case_rmse)),
        p90_rmse=float(np.quantile(case_rmse, .90)),
        worst_rmse=float(case_rmse.max()),
        nonzero_cases=int((np.asarray(weight) > 1e-12).sum()),
    )


probability = np.zeros(len(df))
continuous = np.zeros(len(df))
for train_idx, valid_idx in GroupKFold(5).split(df, groups=df.well):
    train, valid = df.iloc[train_idx], df.iloc[valid_idx]
    scaler = StandardScaler().fit(train[FEATURES])
    x_train, x_valid = scaler.transform(train[FEATURES]), scaler.transform(valid[FEATURES])
    gate = LogisticRegression(C=.5, max_iter=2000, random_state=42).fit(
        x_train, train.beam_helps, sample_weight=train.n_hidden)
    probability[valid_idx] = gate.predict_proba(x_valid)[:, 1]
    model = Ridge(alpha=10.).fit(x_train, train.optimal_weight, sample_weight=train.n_hidden)
    continuous[valid_idx] = np.clip(model.predict(x_valid), 0., .20)

v1 = np.where(df.prefix_rmse <= 8., .15 * np.maximum(0., 1. - df.prefix_rmse / 8.), 0.)
policies = {
    "anchor": np.zeros(len(df)),
    "v1": v1,
    "v1_binary_veto": v1 * (probability >= .5),
    "v1_probability_shrink": v1 * probability,
    "v1_continuous_cap": np.minimum(v1, continuous),
    "v1_double_veto": np.minimum(v1, continuous) * (probability >= .5),
}
summary = pd.DataFrame([{"policy": name, **metrics(weight)} for name, weight in policies.items()]).sort_values("pooled_rmse")
summary.to_csv(OUT / "summary.csv", index=False)
run = {"best_policy": str(summary.iloc[0].policy), "best_pooled_rmse": float(summary.iloc[0].pooled_rmse)}
(OUT / "run.json").write_text(json.dumps(run, indent=2) + "\n")
print(summary.to_string(index=False))
