import csv
import io
from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).with_name("rogii_alignment_lab_v1.ipynb")
RIDGE_SOURCE = OUT.parent.parent / "rogii_candidate_oracle_audit_v1" / "ranker_ridge_coefficients_v4.csv"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# ROGII alignment lab v1

## tl;dr

This diagnostic notebook tests the GR-alignment hypothesis at component level. It uses monotonic slope-constrained DTW on an absolute TVT grid, multi-scale locally normalized emissions, and both typewell and self-log references.

No `submission.csv` is created. The experiment is viable only if prefix-selected alignment improves the frozen Ridge by at least 1 ft pooled RMSE and improves it in every spatial evaluation fold.
"""
)

md(
    """## Context & Methods

The earlier experiment decoded a coarse offset around Ridge. This lab instead decodes an absolute monotonic TVT path. For every fixed emission family, drilling direction is selected only from a visible-prefix backtest; suffix TVT is used only after prediction for audit metrics.

### Key Assumptions

- Validation is pooled per point and grouped by complete well.
- Spatial clusters are evaluation slices, not random rows.
- Ridge is a frozen cross-fitted center imported from the geometry experiment.
- Self-log reference is built only from visible horizontal-well GR and `TVT_input`.
- Long GR gaps receive low emission weight rather than being treated as observed signal.
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
import json, os, time, warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

VERSION = "v1"
SMOKE = os.environ.get("ROGII_SMOKE", "0") == "1"
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
LEGAL_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
STRIDE = 10
TVT_STEP = 2.0
BAND_FT = 120.0
MAX_STEP = 4
CONFIGS = ["type_raw", "type_multiscale", "self_multiscale", "hybrid_multiscale"]


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
    ids = ids[:20]


def rmse(y_true, prediction):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(prediction, float)) ** 2)))


def robust_scale(values):
    values = np.asarray(values, float)
    median = np.nanmedian(values)
    scale = 1.4826 * np.nanmedian(np.abs(values - median))
    return float(max(scale, 1e-3))


print({"version": VERSION, "root": str(ROOT), "wells": len(ids), "smoke": SMOKE})
'''
)

md("## Data - locally normalized sequence features")

code(
    r'''def fill_gr(values):
    original = pd.Series(values, dtype=float)
    filled = original.interpolate(limit=100, limit_direction="both")
    fallback = float(original.median()) if original.notna().any() else 0.0
    return filled.fillna(fallback).to_numpy(float), original.notna().to_numpy(float)


def feature_bank(values):
    values = np.asarray(values, float)
    series = pd.Series(values)
    mean = series.rolling(61, center=True, min_periods=10).mean().bfill().ffill().to_numpy(float)
    std = series.rolling(61, center=True, min_periods=10).std().bfill().ffill().to_numpy(float)
    std = np.maximum(std, robust_scale(values) * 0.25)
    raw_z = np.clip((values - mean) / std, -5.0, 5.0)
    smooth_21 = series.rolling(21, center=True, min_periods=1).mean().to_numpy(float)
    smooth_61 = series.rolling(61, center=True, min_periods=1).mean().to_numpy(float)
    smooth_z = np.clip((smooth_21 - mean) / std, -5.0, 5.0)
    derivative = np.gradient(smooth_21)
    derivative = np.clip((derivative - np.median(derivative)) / robust_scale(derivative), -5.0, 5.0)
    slow_derivative = np.gradient(smooth_61)
    slow_derivative = np.clip((slow_derivative - np.median(slow_derivative)) / robust_scale(slow_derivative), -5.0, 5.0)
    return np.c_[raw_z, smooth_z, derivative, slow_derivative]


def reference_grid(typewell):
    typewell = typewell.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    grouped = typewell.groupby("TVT", as_index=False)["GR"].median()
    lo = np.floor(grouped["TVT"].min() / TVT_STEP) * TVT_STEP
    hi = np.ceil(grouped["TVT"].max() / TVT_STEP) * TVT_STEP
    grid = np.arange(lo, hi + TVT_STEP, TVT_STEP)
    values = np.interp(grid, grouped["TVT"], grouped["GR"])
    return grid, feature_bank(values)


def self_reference(frame, reference_rows, grid):
    part = frame.iloc[reference_rows][["TVT_input", "GR"]].dropna()
    if len(part) < 30:
        return np.zeros((len(grid), 4)), np.zeros(len(grid), bool)
    bins = np.round(part["TVT_input"].to_numpy(float) / TVT_STEP) * TVT_STEP
    grouped = pd.DataFrame({"TVT": bins, "GR": part["GR"].to_numpy(float)}).groupby("TVT", as_index=False)["GR"].median()
    values = np.interp(grid, grouped["TVT"], grouped["GR"])
    coverage = (grid >= grouped["TVT"].min()) & (grid <= grouped["TVT"].max())
    return feature_bank(values), coverage
'''
)

md("## Methods - monotonic slope-constrained DTW")

code(
    r'''def emission_matrix(horizontal_features, sample_rows, reference_features, reliability, ridge, grid, channels):
    difference = horizontal_features[sample_rows][:, None, channels] - reference_features[:, channels][None, :, :]
    cost = np.minimum(difference * difference, 16.0).mean(axis=2)
    cost *= reliability[sample_rows, None]
    cost += 0.04 * ((grid[None, :] - ridge[:, None]) / 30.0) ** 2
    cost[np.abs(grid[None, :] - ridge[:, None]) > BAND_FT] = np.inf
    return cost


def decode_dtw(emission, grid, last_tvt, direction):
    ordered = emission if direction > 0 else emission[:, ::-1]
    ordered_grid = grid if direction > 0 else grid[::-1]
    rows, states = ordered.shape
    back = np.zeros((rows, states), np.int16)
    dp = ordered[0] + 2.0 * ((ordered_grid - last_tvt) / TVT_STEP) ** 2
    for row in range(1, rows):
        new = np.full(states, np.inf)
        source = np.zeros(states, np.int16)
        for step in range(MAX_STEP + 1):
            destination = np.arange(step, states)
            origin = destination - step
            candidate = dp[origin] + 0.08 * step * step
            better = candidate < new[destination]
            new[destination[better]] = candidate[better]
            source[destination[better]] = origin[better]
        dp = new + ordered[row]
        back[row] = source
    state = int(np.argmin(dp))
    path = np.empty(rows, int)
    path[-1] = state
    for row in range(rows - 1, 0, -1):
        state = int(back[row, state])
        path[row - 1] = state
    return ordered_grid[path], float(np.min(dp) / rows)


# Runnable self-check: the decoder recovers an exact increasing synthetic path.
synthetic_grid = np.arange(0.0, 20.0, 1.0)
synthetic_truth = np.arange(2, 10)
synthetic_cost = np.full((len(synthetic_truth), len(synthetic_grid)), 25.0)
synthetic_cost[np.arange(len(synthetic_truth)), synthetic_truth] = 0.0
synthetic_path, _ = decode_dtw(synthetic_cost, synthetic_grid, 2.0, 1)
assert np.array_equal(synthetic_path, synthetic_grid[synthetic_truth])
'''
)

md("## Methods - prefix-only configuration and direction selection")

code(
    r'''def ridge_path(frame, target_rows, anchor_row, anchor_tvt, well_id):
    md_values = frame["MD"].to_numpy(float)
    z = frame["Z"].to_numpy(float)
    span = max(float(md_values[target_rows[-1]] - md_values[anchor_row]), 1.0)
    x = (md_values[target_rows] - md_values[anchor_row]) / span
    base = anchor_tvt - (z[target_rows] - z[anchor_row])
    coefficient = ridge_coefficients.loc[well_id]
    return base + float(coefficient["ridge_c1"]) * x + float(coefficient["ridge_c2"]) * x * x


def all_emissions(frame, reference_rows, target_rows, ridge, grid, type_features):
    horizontal_gr, reliability = fill_gr(frame["GR"])
    horizontal_features = feature_bank(horizontal_gr)
    self_features, self_coverage = self_reference(frame, reference_rows, grid)
    sample_local = np.unique(np.r_[np.arange(0, len(target_rows), STRIDE), len(target_rows) - 1])
    sample_rows = target_rows[sample_local]
    type_raw = emission_matrix(horizontal_features, sample_rows, type_features, reliability, ridge[sample_local], grid, [0])
    type_multi = emission_matrix(horizontal_features, sample_rows, type_features, reliability, ridge[sample_local], grid, [0, 1, 2, 3])
    self_multi = emission_matrix(horizontal_features, sample_rows, self_features, reliability, ridge[sample_local], grid, [0, 1, 2, 3])
    self_multi[:, ~self_coverage] += 8.0
    hybrid = type_multi.copy()
    hybrid[:, self_coverage] = 0.5 * type_multi[:, self_coverage] + 0.5 * self_multi[:, self_coverage]
    return sample_local, {
        "type_raw": type_raw, "type_multiscale": type_multi,
        "self_multiscale": self_multi, "hybrid_multiscale": hybrid,
    }, float(1.0 - reliability[target_rows].mean()), float(self_coverage.mean())


def prefix_backtest(frame, well_id, grid, type_features):
    known_rows = np.flatnonzero(frame["TVT_input"].notna().to_numpy())
    cut = max(30, int(len(known_rows) * 0.70))
    reference_rows, validation_rows = known_rows[:cut], known_rows[cut:]
    if len(validation_rows) < 30:
        return {name: 1 for name in CONFIGS}, CONFIGS[0]
    anchor_row = int(reference_rows[-1])
    anchor_tvt = float(frame.loc[anchor_row, "TVT_input"])
    ridge = ridge_path(frame, validation_rows, anchor_row, anchor_tvt, well_id)
    sample_local, emissions, _, _ = all_emissions(frame, reference_rows, validation_rows, ridge, grid, type_features)
    result, direction_by_config = {}, {}
    truth = frame.loc[validation_rows, "TVT_input"].to_numpy(float)
    for name, emission in emissions.items():
        choices = []
        for direction in [-1, 1]:
            sampled, _ = decode_dtw(emission, grid, anchor_tvt, direction)
            prediction = np.interp(np.arange(len(validation_rows)), sample_local, sampled)
            choices.append((rmse(truth, prediction), direction))
        result[name], direction_by_config[name] = min(choices)
    return direction_by_config, min(result, key=result.get)
'''
)

md("## Results - decode all wells")

code(
    r'''prediction_frames, diagnostic_rows = [], []
start_time = time.time()

for index, well_id in enumerate(ids, 1):
    frame = pd.read_csv(TRAIN_DIR / f"{well_id}{HORIZONTAL_SUFFIX}", usecols=LEGAL_COLUMNS + ["TVT"])
    typewell = pd.read_csv(TRAIN_DIR / f"{well_id}{TYPEWELL_SUFFIX}", usecols=["TVT", "GR"])
    known_rows = np.flatnonzero(frame["TVT_input"].notna().to_numpy())
    target_rows = np.flatnonzero(frame["TVT_input"].isna().to_numpy())
    anchor_row = int(known_rows[-1])
    anchor_tvt = float(frame.loc[anchor_row, "TVT_input"])
    truth = frame.loc[target_rows, "TVT"].to_numpy(float)
    ridge = ridge_path(frame, target_rows, anchor_row, anchor_tvt, well_id)
    grid, type_features = reference_grid(typewell)
    directions, selected_config = prefix_backtest(frame, well_id, grid, type_features)
    sample_local, emissions, missing_share, self_coverage = all_emissions(
        frame, known_rows, target_rows, ridge, grid, type_features
    )
    predictions, path_costs = {"ridge_prior": ridge}, {}
    for name, emission in emissions.items():
        sampled, path_cost = decode_dtw(emission, grid, anchor_tvt, directions[name])
        predictions[name] = np.interp(np.arange(len(target_rows)), sample_local, sampled)
        path_costs[name] = path_cost
    predictions["prefix_selected"] = predictions[selected_config]
    oracle_config = min(CONFIGS, key=lambda name: rmse(truth, predictions[name]))
    predictions["oracle_config"] = predictions[oracle_config]
    prediction_frames.append(pd.DataFrame({
        "id": [f"{well_id}_{row}" for row in target_rows], "well_id": well_id,
        "row_index": target_rows, "target": truth, **predictions,
    }))
    correction = predictions["prefix_selected"] - ridge
    oracle_correction = truth - ridge
    diagnostic_rows.append({
        "well_id": well_id, "rows": len(target_rows), "ps_x": frame.loc[anchor_row, "X"], "ps_y": frame.loc[anchor_row, "Y"],
        "missing_gr_share": missing_share, "self_reference_coverage": self_coverage,
        "prefix_selected_config": selected_config, "oracle_config": oracle_config,
        "prefix_selected_direction": directions[selected_config],
        "correction_correlation": float(np.corrcoef(correction, oracle_correction)[0, 1]) if np.std(correction) > 0 else 0.0,
        "cycle_skip_share": float((np.abs(predictions["prefix_selected"] - truth) > 30.0).mean()),
        **{f"{name}_path_cost": path_costs[name] for name in CONFIGS},
    })
    if index % 100 == 0:
        print("decoded", index, "/", len(ids), "elapsed", round(time.time() - start_time, 1))

OOF = pd.concat(prediction_frames, ignore_index=True)
WELLS = pd.DataFrame(diagnostic_rows)
CANDIDATES = ["ridge_prior", *CONFIGS, "prefix_selected", "oracle_config"]
assert OOF["id"].is_unique and np.isfinite(OOF[CANDIDATES].to_numpy(float)).all()
print("OOF", OOF.shape)
'''
)

md("## Results - pooled and spatial stability")

code(
    r'''coordinates = StandardScaler().fit_transform(WELLS[["ps_x", "ps_y"]])
WELLS["spatial_fold"] = KMeans(n_clusters=min(5, len(WELLS)), random_state=42, n_init=10).fit_predict(coordinates)
fold_map = WELLS.set_index("well_id")["spatial_fold"]
OOF["spatial_fold"] = OOF["well_id"].map(fold_map)

SCORES = pd.DataFrame([
    {"candidate": candidate, "pooled_rmse": rmse(OOF["target"], OOF[candidate])}
    for candidate in CANDIDATES
]).sort_values("pooled_rmse").reset_index(drop=True)
SPATIAL_SCORES = pd.DataFrame([
    {"spatial_fold": fold, "candidate": candidate, "rows": len(group), "pooled_rmse": rmse(group["target"], group[candidate])}
    for fold, group in OOF.groupby("spatial_fold") for candidate in CANDIDATES
])
ridge_score = float(SCORES.set_index("candidate").loc["ridge_prior", "pooled_rmse"])
selected_score = float(SCORES.set_index("candidate").loc["prefix_selected", "pooled_rmse"])
spatial = SPATIAL_SCORES.pivot(index="spatial_fold", columns="candidate", values="pooled_rmse")
viable = bool((ridge_score - selected_score >= 1.0) and (spatial["prefix_selected"] < spatial["ridge_prior"]).all())

for candidate in CANDIDATES:
    per_well = OOF.groupby("well_id", sort=False).apply(
        lambda group: rmse(group["target"], group[candidate]), include_groups=False
    )
    WELLS[f"{candidate}_rmse"] = WELLS["well_id"].map(per_well)

display(SCORES)
display(spatial)
display(WELLS[["missing_gr_share", "self_reference_coverage", "correction_correlation", "cycle_skip_share"]].describe())
print({"viable": viable, "improvement_ft": ridge_score - selected_score})
'''
)

md("## Takeaways and artifacts")

code(
    r'''SCORES.to_csv(WORK / "alignment_lab_scores_v1.csv", index=False)
SPATIAL_SCORES.to_csv(WORK / "alignment_lab_spatial_scores_v1.csv", index=False)
WELLS.to_csv(WORK / "alignment_lab_wells_v1.csv", index=False)
prediction_path = WORK / "alignment_lab_oof_predictions_v1.parquet"
OOF.to_parquet(prediction_path, index=False)

summary = {
    "version": VERSION, "wells": int(OOF["well_id"].nunique()), "rows": int(len(OOF)),
    "stride": STRIDE, "tvt_step": TVT_STEP, "band_ft": BAND_FT, "max_step": MAX_STEP,
    "scores": SCORES.to_dict("records"),
    "ridge_prior_rmse": ridge_score, "prefix_selected_rmse": selected_score,
    "improvement_ft": ridge_score - selected_score,
    "spatial_fold_scores": SPATIAL_SCORES.to_dict("records"),
    "viability_rule": "prefix_selected improves Ridge by >=1 ft and wins in every spatial fold",
    "viable": viable,
    "decision": "promote_to_guarded_inference" if viable else "stop_or_revise_alignment",
    "prediction_artifact": prediction_path.name,
    "caveats": [
        "Spatial clusters are evaluation slices; this per-well decoder fits no cross-well model.",
        "Oracle configuration uses suffix TVT and is diagnostic only.",
        "Monotonic TVT cannot represent genuinely reversing local geology paths.",
        "No submission is created.",
    ],
}
(WORK / "alignment_lab_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
