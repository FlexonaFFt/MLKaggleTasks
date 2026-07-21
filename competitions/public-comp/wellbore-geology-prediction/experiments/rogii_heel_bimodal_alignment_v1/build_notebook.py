import csv
import io
from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).with_name("rogii_heel_bimodal_alignment_v1.ipynb")
RIDGE_SOURCE = OUT.parent.parent / "rogii_candidate_oracle_audit_v1" / "ranker_ridge_coefficients_v4.csv"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# ROGII heel-calibrated aggregation v3

## tl;dr

V1 established a real heel-calibrated offset signal (`13.996`). V2 rejected piecewise Viterbi (`14.516`). V3 cross-fits a well-level aggregator over three independent experts: neighbor (`13.351`), heel, and Ridge fallback.

The aggregator learns only from other folds and emits constant per-well convex weights. No `submission.csv` is created.
"""
)

md(
    """## Context & Methods

### Key Assumptions

- The full horizontal GR trace and typewell are legal inputs.
- Affine GR calibration uses only rows with visible `TVT_input`.
- Candidate TVT paths are centered on the previously cross-fitted Ridge prediction.
- Every correction is exactly zero at projection start.
- Heel search remains bounded to `±60 ft`; rejected Viterbi paths are not executed.
- Validation targets enter only scores and oracle diagnostics.
- Spatially stratified folds are reporting slices; the per-well aligner fits no cross-well model.

Promotion gate: aggregator RMSE below neighbor-transfer `13.351`, wins in at least 4/5 slices, and does not worsen worst-decile SSE share.
"""
)

with RIDGE_SOURCE.open(newline="", encoding="utf-8") as source:
    rows = csv.DictReader(source)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["well_id", "ridge_c1", "ridge_c2"])
    writer.writeheader()
    writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
code("RIDGE_CSV = " + repr(buffer.getvalue()))

code(
    r'''from io import StringIO
from pathlib import Path
import hashlib, json, os, warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

VERSION = "v3"
SEED = 42
N_FOLDS = 5
STRIDE = 5
SHIFT_GRID = np.arange(-60.0, 60.01, 3.0)
SMOKE = os.environ.get("ROGII_SMOKE", "0") == "1"
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
LEGAL_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
PRIMARY = "aggregator_g50"


def find_root():
    roots = [
        Path(os.environ["ROGII_DATA"]) if os.environ.get("ROGII_DATA") else None,
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path.cwd() / "datasets", Path.cwd().parent / "datasets",
    ]
    roots.extend(parent / "datasets" for parent in list(Path.cwd().parents)[:4])
    for root in roots:
        if root is not None and (root / "train").exists() and (root / "test").exists():
            return root
    raise FileNotFoundError("ROGII dataset root not found")


ROOT = find_root()
TRAIN_DIR = ROOT / "train"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
ridge_coefficients = pd.read_csv(StringIO(RIDGE_CSV)).set_index("well_id")
ids = sorted(path.name.removesuffix(HORIZONTAL_SUFFIX) for path in TRAIN_DIR.glob(f"*{HORIZONTAL_SUFFIX}"))
if SMOKE:
    ids = ids[:30]
assert set(ids) <= set(ridge_coefficients.index)


def rmse(actual, prediction):
    return float(np.sqrt(np.mean((np.asarray(actual, float) - np.asarray(prediction, float)) ** 2)))


def filled(values):
    series = pd.Series(values, dtype=float).interpolate(limit_direction="both")
    return series.fillna(float(series.median()) if series.notna().any() else 0.0).to_numpy(float)


def median_smooth(values, window=7):
    return pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy(float)


print({"version": VERSION, "root": str(ROOT), "wells": len(ids), "primary": PRIMARY, "smoke": SMOKE})
'''
)

md("## Data - legal Ridge center and heel calibration")

code(
    r'''def affine_heel_calibration(horizontal_gr, known_idx, tvt_input, tw_tvt, tw_gr):
    reference = np.interp(tvt_input[known_idx], tw_tvt, tw_gr)
    design = np.c_[horizontal_gr[known_idx], np.ones(len(known_idx))]
    coef = np.linalg.lstsq(design, reference, rcond=None)[0]
    for _ in range(2):
        residual = reference - design @ coef
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weight = 1.0 / (1.0 + (residual / (2.0 * scale)) ** 2)
        coef = np.linalg.lstsq(design * weight[:, None], reference * weight, rcond=None)[0]
    gain = float(np.clip(coef[0], 0.2, 5.0))
    bias = float(np.median(reference - gain * horizontal_gr[known_idx]))
    calibrated = gain * horizontal_gr + bias
    residual = reference - calibrated[known_idx]
    sigma = float(np.clip(1.4826 * np.median(np.abs(residual - np.median(residual))), 5.0, 50.0))
    r2 = 1.0 - float(np.sum(residual**2) / max(np.sum((reference - reference.mean())**2), 1e-9))
    return calibrated, sigma, r2, gain, bias


def well_inputs(well_id):
    frame = pd.read_csv(TRAIN_DIR / f"{well_id}{HORIZONTAL_SUFFIX}", usecols=LEGAL_COLUMNS + ["TVT"])
    typewell = pd.read_csv(TRAIN_DIR / f"{well_id}{TYPEWELL_SUFFIX}", usecols=["TVT", "GR"])
    known = frame["TVT_input"].notna().to_numpy()
    known_idx, target_idx = np.flatnonzero(known), np.flatnonzero(~known)
    if len(known_idx) < 20 or len(target_idx) < 20:
        return None
    ps = known_idx[-1]
    md_values = frame["MD"].to_numpy(float)
    z = frame["Z"].to_numpy(float)
    tvt_input = frame["TVT_input"].to_numpy(float)
    span = max(float(md_values[target_idx[-1]] - md_values[ps]), 1.0)
    s = (md_values[target_idx] - md_values[ps]) / span
    base = float(tvt_input[ps]) - (z[target_idx] - z[ps])
    coefficient = ridge_coefficients.loc[well_id]
    ridge = base + float(coefficient["ridge_c1"]) * s + float(coefficient["ridge_c2"]) * s * s
    truth = frame["TVT"].to_numpy(float)[target_idx]
    if not np.isfinite(truth).all():
        return None

    horizontal_gr = filled(frame["GR"])
    typewell = typewell.dropna().sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr = filled(typewell["GR"])
    calibrated, sigma, calibration_r2, gain, bias = affine_heel_calibration(
        horizontal_gr, known_idx, tvt_input, tw_tvt, tw_gr
    )
    return {
        "frame": frame, "target_idx": target_idx, "truth": truth, "ridge": ridge, "s": s,
        "horizontal_gr": calibrated, "tw_tvt": tw_tvt, "tw_gr": tw_gr, "sigma": sigma,
        "calibration_r2": calibration_r2, "calibration_gain": gain, "calibration_bias": bias,
        "ps_x": float(frame.loc[ps, "X"]), "ps_y": float(frame.loc[ps, "Y"]),
    }
'''
)

md("## Methods - bounded two-mode search")

code(
    r'''CONFIGS = {
    "raw_r15": {"denoise": False, "ramp_fraction": 0.15},
    "denoise_r05": {"denoise": True, "ramp_fraction": 0.05},
    "denoise_r15": {"denoise": True, "ramp_fraction": 0.15},
    "denoise_r30": {"denoise": True, "ramp_fraction": 0.30},
}


def anchored_ramp(s, fraction):
    u = np.clip(np.asarray(s, float) / fraction, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def shift_costs(inputs, config):
    target_idx, ridge, s = inputs["target_idx"], inputs["ridge"], inputs["s"]
    local = np.unique(np.r_[np.arange(0, len(target_idx), STRIDE), len(target_idx) - 1])
    rows = target_idx[local]
    ramp = anchored_ramp(s[local], config["ramp_fraction"])
    candidate_tvt = ridge[local, None] + ramp[:, None] * SHIFT_GRID[None, :]
    horizontal = inputs["horizontal_gr"]
    tw_gr = inputs["tw_gr"]
    if config["denoise"]:
        horizontal = median_smooth(horizontal, 7)
        tw_gr = median_smooth(tw_gr, 7)
    expected = np.interp(candidate_tvt.ravel(), inputs["tw_tvt"], tw_gr).reshape(candidate_tvt.shape)
    residual = (horizontal[rows, None] - expected) / inputs["sigma"]
    outside = ((candidate_tvt < inputs["tw_tvt"][0]) | (candidate_tvt > inputs["tw_tvt"][-1])) * 16.0
    return np.mean(np.minimum(residual * residual, 16.0) + outside, axis=0)


def separated_modes(cost, separation=9.0):
    first = int(np.argmin(cost))
    eligible = np.abs(SHIFT_GRID - SHIFT_GRID[first]) >= separation
    second = np.flatnonzero(eligible)[int(np.argmin(cost[eligible]))]
    return first, second


def config_predictions(inputs, config):
    cost = shift_costs(inputs, config)
    first, second = separated_modes(cost)
    shift1, shift2 = float(SHIFT_GRID[first]), float(SHIFT_GRID[second])
    temperature = max(2.0 * float(cost[first]), 0.25)
    logits = -np.array([cost[first], cost[second]]) / temperature
    probability = np.exp(logits - logits.max()); probability /= probability.sum()
    posterior_shift = float(probability @ np.array([shift1, shift2]))
    ramp = anchored_ramp(inputs["s"], config["ramp_fraction"])
    ridge = inputs["ridge"]
    return {
        "best": ridge + shift1 * ramp,
        "posterior": ridge + posterior_shift * ramp,
        "midpoint": ridge + 0.5 * (shift1 + shift2) * ramp,
        "posterior_g10": ridge + 0.10 * posterior_shift * ramp,
        "posterior_g25": ridge + 0.25 * posterior_shift * ramp,
        "posterior_g50": ridge + 0.50 * posterior_shift * ramp,
        "shift1": shift1, "shift2": shift2, "posterior_shift": posterior_shift,
        "probability1": float(probability[0]), "cost1": float(cost[first]), "cost2": float(cost[second]),
    }


PATH_CONFIGS = {
    "path_stiff_g10": {"transition": 1.0, "radius": 1, "center": 0.01, "gain": 0.10},
    "path_smooth_g10": {"transition": 0.25, "radius": 2, "center": 0.005, "gain": 0.10},
    "path_smooth_g25": {"transition": 0.25, "radius": 2, "center": 0.005, "gain": 0.25},
    "path_fault_g10": {"transition": 0.10, "radius": 4, "center": 0.002, "gain": 0.10},
}


def path_emission(inputs):
    target_idx, ridge = inputs["target_idx"], inputs["ridge"]
    local = np.unique(np.r_[np.arange(0, len(target_idx), STRIDE), len(target_idx) - 1])
    rows = target_idx[local]
    candidates = ridge[local, None] + SHIFT_GRID[None, :]
    horizontal = median_smooth(inputs["horizontal_gr"], 7)
    tw_gr = median_smooth(inputs["tw_gr"], 7)
    expected = np.interp(candidates.ravel(), inputs["tw_tvt"], tw_gr).reshape(candidates.shape)
    residual = (horizontal[rows, None] - expected) / inputs["sigma"]
    outside = ((candidates < inputs["tw_tvt"][0]) | (candidates > inputs["tw_tvt"][-1])) * 16.0
    return local, np.minimum(residual * residual, 16.0) + outside


def decode_path(emission, transition, radius, center):
    n_rows, n_states = emission.shape
    back = np.zeros((n_rows, n_states), dtype=np.int16)
    dp = emission[0] + 8.0 * (SHIFT_GRID / 3.0) ** 2
    center_cost = center * (SHIFT_GRID / 30.0) ** 2
    for row in range(1, n_rows):
        new = np.full(n_states, np.inf)
        source = np.zeros(n_states, dtype=np.int16)
        for step in range(-radius, radius + 1):
            destination = np.arange(max(0, step), min(n_states, n_states + step))
            origin = destination - step
            candidate = dp[origin] + transition * step * step
            better = candidate < new[destination]
            new[destination[better]] = candidate[better]
            source[destination[better]] = origin[better]
        dp = new + emission[row] + center_cost
        back[row] = source
    state = int(np.argmin(dp))
    states = np.empty(n_rows, dtype=int); states[-1] = state
    for row in range(n_rows - 1, 0, -1):
        state = int(back[row, state]); states[row - 1] = state
    return SHIFT_GRID[states]


def piecewise_predictions(inputs):
    local, emission = path_emission(inputs)
    ramp = anchored_ramp(inputs["s"], 0.05)
    predictions, diagnostics = {}, {}
    for name, config in PATH_CONFIGS.items():
        sampled = decode_path(emission, config["transition"], config["radius"], config["center"])
        offset = np.interp(np.arange(len(inputs["s"])), local, sampled) * ramp
        predictions[name] = inputs["ridge"] + config["gain"] * offset
        diagnostics[f"{name}_roughness"] = float(np.mean(np.abs(np.diff(offset))))
        diagnostics[f"{name}_boundary_share"] = float(np.mean(np.abs(offset) >= SHIFT_GRID.max()))
    return predictions, diagnostics


# Runnable checks: anchor stays fixed, modes are separated, and zero-emission Viterbi stays at zero.
assert anchored_ramp([0.0, 0.15, 1.0], 0.15)[0] == 0.0
demo_cost = (SHIFT_GRID - 12.0) ** 2
demo_first, demo_second = separated_modes(demo_cost)
assert abs(SHIFT_GRID[demo_first] - SHIFT_GRID[demo_second]) >= 9.0
assert np.max(np.abs(decode_path(np.zeros((8, len(SHIFT_GRID))), 1.0, 1, 0.01))) == 0.0
'''
)

md("## Results - full OOF screen")

code(
    r'''prediction_frames, diagnostics = [], []
for number, well_id in enumerate(ids, 1):
    inputs = well_inputs(well_id)
    if inputs is None:
        continue
    predictions = {"ridge_prior": inputs["ridge"]}
    record = {
        "well_id": well_id, "rows": len(inputs["truth"]), "ps_x": inputs["ps_x"], "ps_y": inputs["ps_y"],
        "calibration_r2": inputs["calibration_r2"], "calibration_gain": inputs["calibration_gain"],
        "calibration_bias": inputs["calibration_bias"], "sigma": inputs["sigma"],
    }
    for name, config in CONFIGS.items():
        result = config_predictions(inputs, config)
        for variant in ["best", "posterior", "midpoint", "posterior_g10", "posterior_g25", "posterior_g50"]:
            predictions[f"{name}_{variant}"] = result[variant]
        for field in ["shift1", "shift2", "posterior_shift", "probability1", "cost1", "cost2"]:
            record[f"{name}_{field}"] = result[field]
    truth = inputs["truth"]
    oracle_name = min(predictions, key=lambda name: rmse(truth, predictions[name]))
    predictions["oracle_config"] = predictions[oracle_name]
    record["oracle_config"] = oracle_name
    prediction_frames.append(pd.DataFrame({
        "id": [f"{well_id}_{row}" for row in inputs["target_idx"]],
        "well_id": well_id, "target": truth, **predictions,
    }))
    diagnostics.append(record)
    if number % 100 == 0:
        print("aligned", number, "/", len(ids))

OOF = pd.concat(prediction_frames, ignore_index=True)
WELLS = pd.DataFrame(diagnostics)
CANDIDATES = [column for column in OOF.columns if column not in {"id", "well_id", "target"}]
assert OOF["well_id"].nunique() == len(WELLS) and np.isfinite(OOF[CANDIDATES].to_numpy(float)).all()
print("OOF", OOF.shape, "wells", len(WELLS))
'''
)

md("## Validation - pooled, tail and spatially stratified slices")

code(
    r'''coordinates = StandardScaler().fit_transform(WELLS[["ps_x", "ps_y"]])
WELLS["region"] = KMeans(n_clusters=min(12, len(WELLS)), random_state=SEED, n_init=10).fit_predict(coordinates)
WELLS["fold"] = -1
for region, group in WELLS.groupby("region"):
    ordered = group.assign(key=group["well_id"].map(lambda value: hashlib.sha1(value.encode()).hexdigest())).sort_values("key")
    WELLS.loc[ordered.index, "fold"] = np.arange(len(ordered)) % N_FOLDS
fold_map = WELLS.set_index("well_id")["fold"]
OOF["fold"] = OOF["well_id"].map(fold_map)

neighbor_paths = list(Path("/kaggle/input").glob("**/neighbor_oof_predictions_v3.parquet"))
diagnostic_paths = list(Path("/kaggle/input").glob("**/neighbor_well_diagnostics_v3.csv"))
assert len(neighbor_paths) == 1 and len(diagnostic_paths) == 1, (neighbor_paths, diagnostic_paths)
neighbor = pd.read_parquet(
    neighbor_paths[0], columns=["id", "target", "hybrid_md_600", "nested_selector"]
).rename(columns={"target": "neighbor_target", "hybrid_md_600": "neighbor_hybrid", "nested_selector": "neighbor_nested"})
OOF = OOF.merge(neighbor, on="id", how="left", validate="one_to_one")
assert OOF["neighbor_hybrid"].notna().all() and np.allclose(OOF["target"], OOF["neighbor_target"])
OOF = OOF.drop(columns="neighbor_target")

neighbor_diagnostics = pd.read_csv(diagnostic_paths[0])[
    ["well_id", "path_distance", "path_angle_difference_deg"]
]
WELLS = WELLS.merge(neighbor_diagnostics, on="well_id", how="left", validate="one_to_one")
assert WELLS[["path_distance", "path_angle_difference_deg"]].notna().all().all()

EXPERTS = ["neighbor_hybrid", "denoise_r30_posterior_g25", "ridge_prior"]
WEIGHT_GRID = np.array([
    (neighbor_weight, heel_weight, 1.0 - neighbor_weight - heel_weight)
    for neighbor_weight in np.arange(0.0, 1.01, 0.25)
    for heel_weight in np.arange(0.0, 1.01 - neighbor_weight, 0.25)
])

feature_rows = []
for well_id, group in OOF.groupby("well_id", sort=False):
    expert_values = group[EXPERTS].to_numpy(float)
    truth = group["target"].to_numpy(float)
    candidates = expert_values @ WEIGHT_GRID.T
    oracle_label = int(np.argmin(np.mean((truth[:, None] - candidates) ** 2, axis=0)))
    well = WELLS.set_index("well_id").loc[well_id]
    feature_rows.append({
        "well_id": well_id, "fold": int(well["fold"]), "rows": len(group), "oracle_label": oracle_label,
        "calibration_r2": well["calibration_r2"], "calibration_gain": well["calibration_gain"],
        "sigma": well["sigma"], "posterior_probability": well["denoise_r30_probability1"],
        "posterior_shift": well["denoise_r30_posterior_shift"],
        "cost_gap": well["denoise_r30_cost2"] - well["denoise_r30_cost1"],
        "path_distance": well["path_distance"], "path_angle": well["path_angle_difference_deg"],
        "neighbor_ridge_rms": float(np.sqrt(np.mean((expert_values[:, 0] - expert_values[:, 2]) ** 2))),
        "heel_ridge_rms": float(np.sqrt(np.mean((expert_values[:, 1] - expert_values[:, 2]) ** 2))),
        "neighbor_heel_rms": float(np.sqrt(np.mean((expert_values[:, 0] - expert_values[:, 1]) ** 2))),
    })
AGGREGATOR_WELLS = pd.DataFrame(feature_rows).set_index("well_id", drop=False)
AGG_FEATURES = [column for column in AGGREGATOR_WELLS if column not in {"well_id", "fold", "rows", "oracle_label"}]

weight_rows = []
for fold in range(N_FOLDS):
    train = AGGREGATOR_WELLS["fold"] != fold
    valid = AGGREGATOR_WELLS["fold"] == fold
    model = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=8, learning_rate=0.04, l2_regularization=5.0, random_state=SEED
    )
    sample_weight = AGGREGATOR_WELLS.loc[train, "rows"] / AGGREGATOR_WELLS.loc[train, "rows"].median()
    model.fit(
        AGGREGATOR_WELLS.loc[train, AGG_FEATURES], AGGREGATOR_WELLS.loc[train, "oracle_label"],
        sample_weight=sample_weight,
    )
    probability = model.predict_proba(AGGREGATOR_WELLS.loc[valid, AGG_FEATURES])
    weights = probability @ WEIGHT_GRID[model.classes_]
    for well_id, values in zip(AGGREGATOR_WELLS.index[valid], weights):
        weight_rows.append({"well_id": well_id, "fold": fold, **{f"weight_{expert}": value for expert, value in zip(EXPERTS, values)}})
AGGREGATOR_WEIGHTS = pd.DataFrame(weight_rows).set_index("well_id")
assert len(AGGREGATOR_WEIGHTS) == len(WELLS) and np.allclose(AGGREGATOR_WEIGHTS.filter(like="weight_").sum(axis=1), 1.0)

raw = np.zeros(len(OOF))
for expert in EXPERTS:
    raw += OOF[expert].to_numpy(float) * OOF["well_id"].map(AGGREGATOR_WEIGHTS[f"weight_{expert}"]).to_numpy(float)
OOF["aggregator_raw"] = raw
OOF["aggregator_g25"] = OOF["neighbor_hybrid"] + 0.25 * (raw - OOF["neighbor_hybrid"])
OOF["aggregator_g50"] = OOF["neighbor_hybrid"] + 0.50 * (raw - OOF["neighbor_hybrid"])

OOF["oracle_aggregator"] = np.nan
for well_id, group in OOF.groupby("well_id", sort=False):
    label = int(AGGREGATOR_WELLS.loc[well_id, "oracle_label"])
    OOF.loc[group.index, "oracle_aggregator"] = group[EXPERTS].to_numpy(float) @ WEIGHT_GRID[label]
CANDIDATES = [column for column in OOF.columns if column not in {"id", "well_id", "target", "fold"}]
assert np.isfinite(OOF[CANDIDATES].to_numpy(float)).all()


def candidate_stats(candidate):
    squared = (OOF["target"].to_numpy(float) - OOF[candidate].to_numpy(float)) ** 2
    per_well = OOF.assign(squared=squared).groupby("well_id")["squared"].agg(["sum", "mean"])
    worst = per_well.nlargest(max(1, len(per_well) // 10), "mean")
    return {
        "candidate": candidate, "pooled_rmse": float(np.sqrt(squared.mean())),
        "median_well_rmse": float(np.sqrt(per_well["mean"]).median()),
        "p90_well_rmse": float(np.sqrt(per_well["mean"]).quantile(0.9)),
        "worst10_sse_share": float(worst["sum"].sum() / per_well["sum"].sum()),
    }


SCORES = pd.DataFrame([candidate_stats(candidate) for candidate in CANDIDATES]).sort_values("pooled_rmse")
FOLD_SCORES = pd.DataFrame([
    {"fold": fold, "candidate": candidate, "rows": len(group), "pooled_rmse": rmse(group["target"], group[candidate])}
    for fold, group in OOF.groupby("fold") for candidate in CANDIDATES
])
score_index = SCORES.set_index("candidate")
ridge_score = float(score_index.loc["ridge_prior", "pooled_rmse"])
primary_score = float(score_index.loc[PRIMARY, "pooled_rmse"])
heel_score = float(score_index.loc["denoise_r15_posterior_g25", "pooled_rmse"])
neighbor_score = float(score_index.loc["neighbor_hybrid", "pooled_rmse"])
neighbor_tail = float(score_index.loc["neighbor_hybrid", "worst10_sse_share"])
primary_tail = float(score_index.loc[PRIMARY, "worst10_sse_share"])
pivot = FOLD_SCORES.pivot(index="fold", columns="candidate", values="pooled_rmse")
fold_wins = int((pivot[PRIMARY] < pivot["neighbor_hybrid"]).sum())
viable = bool(primary_score < neighbor_score and fold_wins >= 4 and primary_tail <= neighbor_tail)

display(SCORES.head(15))
display(pivot[["ridge_prior", "neighbor_hybrid", "denoise_r30_posterior_g25", PRIMARY, "oracle_aggregator", "oracle_config"]])
display(AGGREGATOR_WEIGHTS.filter(like="weight_").describe())
display(WELLS[["calibration_r2", "calibration_gain", "sigma", "denoise_r15_probability1", "denoise_r15_posterior_shift"]].describe())
print({"primary": PRIMARY, "vs_neighbor_ft": neighbor_score - primary_score, "fold_wins": fold_wins, "viable": viable})
'''
)

md("## Takeaways and artifacts")

code(
    r'''SCORES.to_csv(WORK / "heel_bimodal_scores_v3.csv", index=False)
FOLD_SCORES.to_csv(WORK / "heel_bimodal_fold_scores_v3.csv", index=False)
AGGREGATOR_WELLS.reset_index(drop=True).to_csv(WORK / "heel_bimodal_aggregator_wells_v3.csv", index=False)
AGGREGATOR_WEIGHTS.reset_index().to_csv(WORK / "heel_bimodal_aggregator_weights_v3.csv", index=False)
WELLS.to_csv(WORK / "heel_bimodal_wells_v3.csv", index=False)
OOF.to_parquet(WORK / "heel_bimodal_oof_v3.parquet", index=False)
summary = {
    "version": VERSION, "wells": int(WELLS.shape[0]), "rows": int(len(OOF)), "primary": PRIMARY,
    "ridge_rmse": ridge_score, "heel_v1_rmse": heel_score, "neighbor_rmse": neighbor_score,
    "primary_rmse": primary_score, "improvement_vs_neighbor_ft": neighbor_score - primary_score,
    "neighbor_worst10_sse_share": neighbor_tail, "primary_worst10_sse_share": primary_tail,
    "fold_wins_vs_neighbor": fold_wins, "viable": viable,
    "mean_weights": AGGREGATOR_WEIGHTS.filter(like="weight_").mean().to_dict(),
    "scores": SCORES.head(15).to_dict("records"),
    "viability_rule": "aggregator_g50 beats neighbor_hybrid, wins >=4/5 slices, and does not worsen worst10 SSE share",
    "decision": "build_submission_pipeline" if viable else "stop_or_revise_aggregator",
    "caveats": [
        "Ridge center is imported from an earlier cross-fitted random-well experiment.",
        "Neighbor predictions are imported from their original cross-fitted V3 artifact.",
        "Aggregator labels use only outer-training wells and predictions are outer-fold held out.",
        "V1 heel configuration was selected in a preceding experiment on the same wells.",
        "No submission is created.",
    ],
}
(WORK / "heel_bimodal_summary_v3.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
