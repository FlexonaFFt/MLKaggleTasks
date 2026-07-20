from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).with_name("rogii_candidate_oracle_audit_v1.ipynb")
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# ROGII cross-fitted sequence ranker v4

This is version 4 of the same diagnostic Kaggle notebook. It never creates `submission.csv`.

V3 found strong ordinal signal in multi-scale GR/typewell cost (median Spearman **0.807**) but direct argmin degraded RMSE. V4 tests the next concrete hypothesis:

> Can a cross-fitted calibrator turn that cost signal into a selector that improves the OOF Ridge prior?

Prediction family:

`TVT_hat = TVT_PS - (Z - Z_PS) + c1*x + c2*x^2`, where `x=0` at PS and `x=1` at the toe.

Two selectors are evaluated: a one-parameter distance regularizer tuned on other folds, and a small histogram gradient-boosting ranker trained on other folds. Validation suffix `TVT` is used only for held-out scoring.
"""
)

md(
    """## Experiment contract

- The organizer's actual `TVT_input` prefix/suffix mask is preserved.
- Five folds are split by complete well.
- Oracle coefficients from validation suffix targets are used only for scoring.
- The Ridge coefficient prior trains only on other folds.
- A 7x7 coefficient grid uses offsets `[-120,-60,-30,0,30,60,120]` around that prior.
- Raw, rolling-21, and rolling-61 GR alignment costs are evaluated on a stride-10 suffix sample.
- Candidate MSE labels for every validation well are invisible to both selectors fitted for that fold.
- Primary metric is pooled per-point RMSE.

Artifacts: `ranker_scores_v4.csv`, `ranker_well_diagnostics_v4.csv`, `ranker_selected_coefficients_v4.csv`, `ranker_summary_v4.json`, and compressed OOF predictions.
"""
)

code(
    r'''from pathlib import Path
import json, os, time, warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

VERSION = "v4"
SEED = 42
N_SPLITS = 5
SMOKE = os.environ.get("ROGII_SMOKE", "0") == "1"
LEGAL_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
GRID_OFFSETS = np.array([-120., -60., -30., 0., 30., 60., 120.])


def find_root():
    roots = [
        Path(os.environ["ROGII_DATA"]) if os.environ.get("ROGII_DATA") else None,
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path.cwd() / "datasets",
        Path.cwd().parent / "datasets",
    ]
    roots.extend(parent / "datasets" for parent in list(Path.cwd().parents)[:4])
    for root in roots:
        if root is not None and (root / "train").exists() and (root / "test").exists():
            return root
    raise FileNotFoundError("ROGII dataset root not found")


ROOT = find_root()
TRAIN_DIR = ROOT / "train"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()


def well_ids():
    return sorted(path.name.removesuffix(HORIZONTAL_SUFFIX) for path in TRAIN_DIR.glob(f"*{HORIZONTAL_SUFFIX}"))


def load_well(well_id):
    return pd.read_csv(TRAIN_DIR / f"{well_id}{HORIZONTAL_SUFFIX}", usecols=LEGAL_COLUMNS + ["TVT"])


def load_typewell(well_id):
    frame = pd.read_csv(TRAIN_DIR / f"{well_id}{TYPEWELL_SUFFIX}")
    return frame[["TVT", "GR"]].dropna().sort_values("TVT").drop_duplicates("TVT")


def split_mask(frame):
    known = frame["TVT_input"].notna().to_numpy()
    return known, ~known


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


ids = well_ids()
if SMOKE:
    ids = ids[:20]
    N_SPLITS = 2

rng = np.random.default_rng(SEED)
shuffled = np.array(ids, dtype=object)
rng.shuffle(shuffled)
FOLD_BY_WELL = {well_id: int(index % N_SPLITS) for index, well_id in enumerate(shuffled)}

print({"version": VERSION, "root": str(ROOT), "wells": len(ids), "folds": N_SPLITS, "smoke": SMOKE})
'''
)

md("## Per-well legal features and oracle coefficient targets")

code(
    r'''def safe_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) < 1e-9:
        return 0.0
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def anchored_coefficients(error, x, degree):
    design = np.column_stack([x ** power for power in range(1, degree + 1)])
    return np.linalg.lstsq(design, error, rcond=None)[0]


def describe_well(well_id):
    full = load_well(well_id)
    target_values = full["TVT"].to_numpy(float)
    frame = full[LEGAL_COLUMNS].copy()
    known, target = split_mask(frame)
    known_idx = np.flatnonzero(known)
    target_idx = np.flatnonzero(target)
    if len(known_idx) < 20 or len(target_idx) < 20 or not np.isfinite(target_values[target_idx]).all():
        return None

    md = frame["MD"].to_numpy(float)
    x_coord = frame["X"].to_numpy(float)
    y_coord = frame["Y"].to_numpy(float)
    z = frame["Z"].to_numpy(float)
    tvt_input = frame["TVT_input"].to_numpy(float)
    ps = known_idx[-1]
    span = max(float(md[target_idx[-1]] - md[ps]), 1.0)
    x = (md[target_idx] - md[ps]) / span
    base = float(tvt_input[ps]) - (z[target_idx] - z[ps])
    truth = target_values[target_idx]
    error = truth - base

    linear = anchored_coefficients(error, x, 1)
    quadratic = anchored_coefficients(error, x, 2)
    cubic = anchored_coefficients(error, x, 3)

    u = tvt_input[known_idx] + z[known_idx]
    prefix_span = max(float(md[ps] - md[known_idx[0]]), 1.0)
    prefix_x = (md[known_idx] - md[ps]) / prefix_span
    prefix_u = u - u[-1]
    prefix_poly = anchored_coefficients(prefix_u, prefix_x, 2)

    slopes = {}
    for window in [100, 300, 800]:
        idx = known_idx[-min(window, len(known_idx)):]
        slopes[f"u_slope_{window}"] = safe_slope(md[idx], tvt_input[idx] + z[idx])
    slopes["u_slope_all"] = safe_slope(md[known_idx], u)
    prefix_slope = float(np.median(list(slopes.values())))

    gr = frame["GR"].astype(float).interpolate(limit_direction="both")
    gr = gr.fillna(float(gr.median()) if gr.notna().any() else 0.0).to_numpy(float)
    dx = float(x_coord[-1] - x_coord[0])
    dy = float(y_coord[-1] - y_coord[0])
    future_base = base - float(tvt_input[ps])
    future_z_poly = anchored_coefficients(future_base, x, 3)

    features = {
        "well_id": well_id,
        "fold": FOLD_BY_WELL[well_id],
        "known_rows": len(known_idx),
        "suffix_rows": len(target_idx),
        "prefix_md_span": prefix_span,
        "suffix_md_span": span,
        "prediction_share": len(target_idx) / len(frame),
        "tvt_ps": float(tvt_input[ps]),
        "z_ps": float(z[ps]),
        "u_ps": float(tvt_input[ps] + z[ps]),
        "prefix_u_range": float(np.ptp(u)),
        "prefix_u_std": float(np.std(u)),
        "prefix_u_c1": float(prefix_poly[0]),
        "prefix_u_c2": float(prefix_poly[1]),
        "prefix_slope_end_correction": prefix_slope * span,
        **slopes,
        "future_dz": float(z[-1] - z[ps]),
        "future_z_range": float(np.ptp(z[target_idx])),
        "future_z_c1": float(future_z_poly[0]),
        "future_z_c2": float(future_z_poly[1]),
        "future_z_c3": float(future_z_poly[2]),
        "future_dx": float(x_coord[-1] - x_coord[ps]),
        "future_dy": float(y_coord[-1] - y_coord[ps]),
        "future_dxy": float(np.hypot(x_coord[-1] - x_coord[ps], y_coord[-1] - y_coord[ps])),
        "azimuth_sin": float(np.sin(np.arctan2(dy, dx))),
        "azimuth_cos": float(np.cos(np.arctan2(dy, dx))),
        "prefix_gr_mean": float(np.mean(gr[known_idx])),
        "prefix_gr_std": float(np.std(gr[known_idx])),
        "suffix_gr_mean": float(np.mean(gr[target_idx])),
        "suffix_gr_std": float(np.std(gr[target_idx])),
        "gr_mean_change": float(np.mean(gr[target_idx]) - np.mean(gr[known_idx])),
        "gr_std_change": float(np.std(gr[target_idx]) - np.std(gr[known_idx])),
        "gr_missing_prefix": float(frame["GR"].iloc[known_idx].isna().mean()),
        "gr_missing_suffix": float(frame["GR"].iloc[target_idx].isna().mean()),
        "oracle_linear_c1": float(linear[0]),
        "oracle_quadratic_c1": float(quadratic[0]),
        "oracle_quadratic_c2": float(quadratic[1]),
        "oracle_cubic_c1": float(cubic[0]),
        "oracle_cubic_c2": float(cubic[1]),
        "oracle_cubic_c3": float(cubic[2]),
    }
    return features


records = []
for index, well_id in enumerate(ids, 1):
    record = describe_well(well_id)
    if record is not None:
        records.append(record)
    if index % 100 == 0:
        print("described", index, "/", len(ids))

WELLS = pd.DataFrame(records).set_index("well_id", drop=False)
TARGET_COLUMNS = ["oracle_quadratic_c1", "oracle_quadratic_c2"]
LINEAR_TARGET = "oracle_linear_c1"
FEATURE_COLUMNS = [
    column for column in WELLS.columns
    if column not in {"well_id", "fold", LINEAR_TARGET, *TARGET_COLUMNS, "oracle_cubic_c1", "oracle_cubic_c2", "oracle_cubic_c3"}
]

assert not any(column.startswith("oracle_") for column in FEATURE_COLUMNS)
assert WELLS[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).notna().all().all()
print("usable wells", len(WELLS), "features", len(FEATURE_COLUMNS))
display(WELLS[TARGET_COLUMNS + [LINEAR_TARGET]].describe().T)
'''
)

md("## Cross-fitted Ridge coefficient prior")

code(
    r'''coefficient_rows = []

for fold in range(N_SPLITS):
    train = WELLS[WELLS["fold"] != fold]
    valid = WELLS[WELLS["fold"] == fold]
    feature_median = train[FEATURE_COLUMNS].median()
    x_train = train[FEATURE_COLUMNS].fillna(feature_median).to_numpy(float)
    x_valid = valid[FEATURE_COLUMNS].fillna(feature_median).to_numpy(float)
    y_train = train[TARGET_COLUMNS].to_numpy(float)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    ridge.fit(x_train, y_train)
    ridge_prediction = ridge.predict(x_valid)

    lower = np.quantile(y_train, 0.01, axis=0)
    upper = np.quantile(y_train, 0.99, axis=0)
    ridge_prediction = np.clip(ridge_prediction, lower, upper)

    for position, well_id in enumerate(valid.index):
        coefficient_rows.append({
            "well_id": well_id,
            "fold": fold,
            "true_c1": float(valid.loc[well_id, TARGET_COLUMNS[0]]),
            "true_c2": float(valid.loc[well_id, TARGET_COLUMNS[1]]),
            "ridge_c1": float(ridge_prediction[position, 0]),
            "ridge_c2": float(ridge_prediction[position, 1]),
        })
    print("fold", fold, "train", len(train), "valid", len(valid))

COEFFICIENTS = pd.DataFrame(coefficient_rows).set_index("well_id", drop=False)
display(COEFFICIENTS.describe().T)
'''
)

md("## Target-free sequence cost and 49-path grid")

code(
    r'''def filled_gr(series):
    values = pd.Series(series, dtype=float).interpolate(limit_direction="both")
    fallback = float(values.median()) if values.notna().any() else 0.0
    return values.fillna(fallback).to_numpy(float)


def rolling(values, window):
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy(float)


def calibrated_typewell(well_id, horizontal, known_idx):
    typewell = load_typewell(well_id)
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr = filled_gr(typewell["GR"])
    tvt_input = horizontal["TVT_input"].to_numpy(float)
    horizontal_gr = filled_gr(horizontal["GR"])
    calibration_idx = known_idx[np.isfinite(tvt_input[known_idx])]
    reference = np.interp(tvt_input[calibration_idx], tw_tvt, tw_gr)
    design = np.c_[reference, np.ones(len(reference))]
    scale, bias = np.linalg.lstsq(design, horizontal_gr[calibration_idx], rcond=None)[0]
    scale = float(np.clip(scale, 0.3, 3.0))
    calibrated = scale * tw_gr + float(bias)
    residual = horizontal_gr[calibration_idx] - np.interp(tvt_input[calibration_idx], tw_tvt, calibrated)
    sigma = float(np.clip(1.4826 * np.median(np.abs(residual - np.median(residual))), 8.0, 50.0))
    return tw_tvt, calibrated, horizontal_gr, scale, float(bias), sigma


def alignment_cost(candidate_tvt, horizontal_gr, sample_idx, tw_tvt, tw_gr, sigma):
    costs = []
    for window in [1, 21, 61]:
        horizontal_scale = horizontal_gr if window == 1 else rolling(horizontal_gr, window)
        typewell_scale = tw_gr if window == 1 else rolling(tw_gr, window)
        reference = np.interp(candidate_tvt.ravel(), tw_tvt, typewell_scale).reshape(candidate_tvt.shape)
        residual = (horizontal_scale[sample_idx, None] - reference) / sigma
        costs.append(np.mean(np.minimum(residual * residual, 16.0), axis=0))
    outside = np.mean((candidate_tvt < tw_tvt[0]) | (candidate_tvt > tw_tvt[-1]), axis=0)
    return costs[0] + 25.0 * outside, np.mean(costs, axis=0) + 25.0 * outside


prediction_frames = []
diagnostic_rows = []
selected_rows = []
candidate_frames = []
start_time = time.time()

for index, well_id in enumerate(WELLS.index, 1):
    full = load_well(well_id)
    target_values = full["TVT"].to_numpy(float)
    frame = full[LEGAL_COLUMNS]
    known, target = split_mask(frame)
    known_idx = np.flatnonzero(known)
    target_idx = np.flatnonzero(target)
    ps = known_idx[-1]
    md_values = frame["MD"].to_numpy(float)
    z = frame["Z"].to_numpy(float)
    tvt_input = frame["TVT_input"].to_numpy(float)
    span = max(float(md_values[target_idx[-1]] - md_values[ps]), 1.0)
    x = (md_values[target_idx] - md_values[ps]) / span
    base = float(tvt_input[ps]) - (z[target_idx] - z[ps])
    truth = target_values[target_idx]
    prior = COEFFICIENTS.loc[well_id]

    offset_c1, offset_c2 = np.meshgrid(GRID_OFFSETS, GRID_OFFSETS, indexing="ij")
    grid_c1 = prior["ridge_c1"] + offset_c1.ravel()
    grid_c2 = prior["ridge_c2"] + offset_c2.ravel()
    sample_local = np.unique(np.r_[np.arange(0, len(target_idx), 10), len(target_idx) - 1])
    sample_idx = target_idx[sample_local]
    sample_paths = base[sample_local, None] + x[sample_local, None] * grid_c1 + x[sample_local, None] ** 2 * grid_c2

    tw_tvt, tw_gr, horizontal_gr, scale, bias, sigma = calibrated_typewell(well_id, frame, known_idx)
    raw_cost, multiscale_cost = alignment_cost(sample_paths, horizontal_gr, sample_idx, tw_tvt, tw_gr, sigma)
    full_paths = base[:, None] + x[:, None] * grid_c1 + x[:, None] ** 2 * grid_c2
    candidate_rmse = np.sqrt(np.mean((truth[:, None] - full_paths) ** 2, axis=0))
    cost_scaled = (multiscale_cost - multiscale_cost.min()) / (multiscale_cost.std() + 1e-8)
    raw_scaled = (raw_cost - raw_cost.min()) / (raw_cost.std() + 1e-8)
    cost_rank = np.argsort(np.argsort(multiscale_cost)).astype(float) / (len(multiscale_cost) - 1)
    candidate_frames.append(pd.DataFrame({
        "well_id": well_id, "fold": int(prior["fold"]), "suffix_rows": len(target_idx),
        "c1": grid_c1, "c2": grid_c2,
        "offset_c1": offset_c1.ravel(), "offset_c2": offset_c2.ravel(),
        "distance_sq": (offset_c1.ravel() / 120.0) ** 2 + (offset_c2.ravel() / 120.0) ** 2,
        "raw_cost": raw_cost, "multiscale_cost": multiscale_cost,
        "raw_scaled": raw_scaled, "cost_scaled": cost_scaled, "cost_rank": cost_rank,
        "prior_c1": float(prior["ridge_c1"]), "prior_c2": float(prior["ridge_c2"]),
        "typewell_scale": scale, "typewell_bias": bias, "robust_sigma": sigma,
        "candidate_mse": candidate_rmse ** 2,
    }))

    raw_index = int(np.argmin(raw_cost))
    multiscale_index = int(np.argmin(multiscale_cost))
    oracle_index = int(np.argmin(candidate_rmse))
    top3_index = np.argsort(multiscale_cost)[:3]
    choices = {
        "ridge_prior": (float(prior["ridge_c1"]), float(prior["ridge_c2"])),
        "raw_selector": (float(grid_c1[raw_index]), float(grid_c2[raw_index])),
        "multiscale_selector": (float(grid_c1[multiscale_index]), float(grid_c2[multiscale_index])),
        "top3_multiscale": (float(np.mean(grid_c1[top3_index])), float(np.mean(grid_c2[top3_index]))),
        "grid_oracle": (float(grid_c1[oracle_index]), float(grid_c2[oracle_index])),
    }
    rank_correlation = spearmanr(multiscale_cost, candidate_rmse).statistic
    oracle_cost_rank = int(np.argsort(np.argsort(multiscale_cost))[oracle_index] + 1)
    diagnostic_rows.append({
        "well_id": well_id, "fold": int(prior["fold"]), "suffix_rows": len(target_idx),
        "cost_rmse_spearman": float(rank_correlation) if np.isfinite(rank_correlation) else 0.0,
        "oracle_cost_rank": oracle_cost_rank, "grid_candidates": len(grid_c1),
        "typewell_scale": scale, "typewell_bias": bias, "robust_sigma": sigma,
        "continuous_oracle_in_grid": bool(abs(prior["true_c1"] - prior["ridge_c1"]) <= 120 and abs(prior["true_c2"] - prior["ridge_c2"]) <= 120),
    })
    predictions = {name: base + c1 * x + c2 * x * x for name, (c1, c2) in choices.items()}
    predictions["last_known"] = np.full(len(target_idx), float(tvt_input[ps]))
    prediction_frames.append(pd.DataFrame({
        "id": [f"{well_id}_{row}" for row in target_idx], "well_id": well_id,
        "row_index": target_idx, "fold": int(prior["fold"]), "target": truth,
        "z_anchor": base, "normalized_md": x,
        **predictions,
    }))
    for name, (c1, c2) in choices.items():
        selected_rows.append({"well_id": well_id, "selector": name, "c1": c1, "c2": c2})
    if index % 100 == 0:
        print("ranked", index, "/", len(WELLS), "elapsed", round(time.time() - start_time, 1))

OOF = pd.concat(prediction_frames, ignore_index=True)
CANDIDATE_TABLE = pd.concat(candidate_frames, ignore_index=True)
WELL_DIAGNOSTICS = pd.DataFrame(diagnostic_rows)
SELECTED = pd.DataFrame(selected_rows)
CANDIDATES = ["last_known", "ridge_prior", "raw_selector", "multiscale_selector", "top3_multiscale", "grid_oracle"]
assert OOF["id"].is_unique and np.isfinite(OOF[CANDIDATES].to_numpy(float)).all()
print("OOF", OOF.shape, "candidate table", CANDIDATE_TABLE.shape)
'''
)

md("## Cross-fitted cost calibration")

code(
    r'''RANKER_FEATURES = [
    "offset_c1", "offset_c2", "distance_sq",
    "raw_scaled", "cost_scaled", "cost_rank",
    "prior_c1", "prior_c2", "typewell_scale", "typewell_bias", "robust_sigma", "suffix_rows",
]
LAMBDA_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
meta_selected_rows = []
fold_rows = []

for fold in range(N_SPLITS):
    train = CANDIDATE_TABLE[CANDIDATE_TABLE["fold"] != fold].copy()
    valid = CANDIDATE_TABLE[CANDIDATE_TABLE["fold"] == fold].copy()
    assert set(train["well_id"]).isdisjoint(valid["well_id"])

    lambda_losses = {}
    for value in LAMBDA_GRID:
        train["selection_score"] = train["cost_scaled"] + value * train["distance_sq"]
        chosen = train.loc[train.groupby("well_id")["selection_score"].idxmin()]
        lambda_losses[value] = float(np.average(chosen["candidate_mse"], weights=chosen["suffix_rows"]))
    best_lambda = min(lambda_losses, key=lambda_losses.get)
    valid["regularized_score"] = valid["cost_scaled"] + best_lambda * valid["distance_sq"]
    regularized = valid.loc[valid.groupby("well_id")["regularized_score"].idxmin()]

    ranker = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=120, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=5.0, random_state=SEED + fold,
    )
    ranker.fit(
        train[RANKER_FEATURES], np.log1p(train["candidate_mse"]),
        sample_weight=train["suffix_rows"],
    )
    valid["ranker_score"] = ranker.predict(valid[RANKER_FEATURES])
    ranked = valid.loc[valid.groupby("well_id")["ranker_score"].idxmin()]

    for selector, chosen in [("regularized_selector", regularized), ("ranker_selector", ranked)]:
        for row in chosen.itertuples(index=False):
            meta_selected_rows.append({
                "well_id": row.well_id, "selector": selector,
                "c1": float(row.c1), "c2": float(row.c2), "fold": fold,
            })
    fold_rows.append({
        "fold": fold, "best_lambda": best_lambda,
        "regularized_train_rmse": float(np.sqrt(lambda_losses[best_lambda])),
    })
    print("calibrated fold", fold, "lambda", best_lambda)

META_SELECTED = pd.DataFrame(meta_selected_rows)
SELECTED = pd.concat([SELECTED, META_SELECTED.drop(columns="fold")], ignore_index=True)
for selector in ["regularized_selector", "ranker_selector"]:
    chosen = META_SELECTED[META_SELECTED["selector"] == selector].set_index("well_id")
    c1 = OOF["well_id"].map(chosen["c1"]).to_numpy(float)
    c2 = OOF["well_id"].map(chosen["c2"]).to_numpy(float)
    OOF[selector] = OOF["z_anchor"] + c1 * OOF["normalized_md"] + c2 * OOF["normalized_md"] ** 2

FOLD_CALIBRATION = pd.DataFrame(fold_rows)
CANDIDATES = [
    "last_known", "ridge_prior", "regularized_selector", "ranker_selector",
    "top3_multiscale", "multiscale_selector", "raw_selector", "grid_oracle",
]
assert META_SELECTED.groupby("selector")["well_id"].nunique().eq(len(WELLS)).all()
assert np.isfinite(OOF[CANDIDATES].to_numpy(float)).all()
display(FOLD_CALIBRATION)
'''
)

md("## Results and decision diagnostics")

code(
    r'''SCORES = pd.DataFrame([
    {"candidate": candidate, "pooled_rmse": rmse(OOF["target"], OOF[candidate])}
    for candidate in CANDIDATES
]).sort_values("pooled_rmse").reset_index(drop=True)

well_rmse = []
for well_id, group in OOF.groupby("well_id", sort=False):
    row = {"well_id": well_id}
    for candidate in CANDIDATES:
        row[f"{candidate}_rmse"] = rmse(group["target"], group[candidate])
    well_rmse.append(row)
WELL_DIAGNOSTICS = WELL_DIAGNOSTICS.merge(pd.DataFrame(well_rmse), on="well_id", validate="one_to_one")
WELL_DIAGNOSTICS["multiscale_regret"] = WELL_DIAGNOSTICS["multiscale_selector_rmse"] - WELL_DIAGNOSTICS["grid_oracle_rmse"]
WELL_DIAGNOSTICS["ranker_delta_vs_prior"] = WELL_DIAGNOSTICS["ranker_selector_rmse"] - WELL_DIAGNOSTICS["ridge_prior_rmse"]

continuous_sse = 0.0
for well_id, group in OOF.groupby("well_id", sort=False):
    coefficient = COEFFICIENTS.loc[well_id]
    full = load_well(well_id)
    known, target = split_mask(full)
    known_idx, target_idx = np.flatnonzero(known), np.flatnonzero(target)
    ps = known_idx[-1]
    md_values, z = full["MD"].to_numpy(float), full["Z"].to_numpy(float)
    span = max(float(md_values[target_idx[-1]] - md_values[ps]), 1.0)
    x = (md_values[target_idx] - md_values[ps]) / span
    base = float(full["TVT_input"].iloc[ps]) - (z[target_idx] - z[ps])
    prediction = base + coefficient["true_c1"] * x + coefficient["true_c2"] * x * x
    continuous_sse += float(np.sum((full["TVT"].to_numpy(float)[target_idx] - prediction) ** 2))
CONTINUOUS_ORACLE_RMSE = float(np.sqrt(continuous_sse / len(OOF)))

display(SCORES)
display(WELL_DIAGNOSTICS[["cost_rmse_spearman", "oracle_cost_rank", "multiscale_regret", "ranker_delta_vs_prior"]].describe())
print("continuous quadratic oracle", CONTINUOUS_ORACLE_RMSE)
'''
)

md("## Artifacts")

code(
    r'''SCORES.to_csv(WORK / "ranker_scores_v4.csv", index=False)
WELL_DIAGNOSTICS.to_csv(WORK / "ranker_well_diagnostics_v4.csv", index=False)
SELECTED.to_csv(WORK / "ranker_selected_coefficients_v4.csv", index=False)
COEFFICIENTS.reset_index(drop=True).to_csv(WORK / "ranker_ridge_coefficients_v4.csv", index=False)
FOLD_CALIBRATION.to_csv(WORK / "ranker_fold_calibration_v4.csv", index=False)
CANDIDATE_TABLE.to_parquet(WORK / "ranker_candidate_table_v4.parquet", index=False)

prediction_path = WORK / "ranker_oof_predictions_v4.parquet"
try:
    OOF.to_parquet(prediction_path, index=False)
except Exception as error:
    prediction_path = WORK / "ranker_oof_predictions_v4.csv.gz"
    OOF.to_csv(prediction_path, index=False, compression="gzip")
    print("parquet fallback", error)

summary = {
    "version": VERSION, "legal_columns": LEGAL_COLUMNS,
    "wells": int(OOF["well_id"].nunique()), "rows": int(len(OOF)),
    "grid_offsets": GRID_OFFSETS.tolist(), "scores": SCORES.to_dict("records"),
    "continuous_quadratic_oracle_rmse": CONTINUOUS_ORACLE_RMSE,
    "median_cost_rmse_spearman": float(WELL_DIAGNOSTICS["cost_rmse_spearman"].median()),
    "mean_cost_rmse_spearman": float(WELL_DIAGNOSTICS["cost_rmse_spearman"].mean()),
    "median_oracle_cost_rank": float(WELL_DIAGNOSTICS["oracle_cost_rank"].median()),
    "grid_coverage_share": float(WELL_DIAGNOSTICS["continuous_oracle_in_grid"].mean()),
    "ranker_better_well_share": float((WELL_DIAGNOSTICS["ranker_delta_vs_prior"] < 0).mean()),
    "fold_calibration": FOLD_CALIBRATION.to_dict("records"),
    "prediction_artifact": prediction_path.name,
}
(WORK / "ranker_summary_v4.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
'''
)


notebook = nbf.v4.new_notebook()
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
notebook.cells = cells
OUT.write_text(nbf.writes(notebook), encoding="utf-8")
print(OUT)
