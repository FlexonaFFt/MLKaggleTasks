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
    """# ROGII heel-calibrated bimodal alignment v1

## tl;dr

Stage 1 tests one new deployable signal: calibrate horizontal GR to its typewell using only the visible prefix, search bounded anchored corrections around the frozen OOF Ridge path, preserve two separated modes, and emit their posterior mean.

The primary configuration is frozen before scoring: median-7 denoise, 15% anchor ramp, posterior correction gain 0.25. No `submission.csv` is created.
"""
)

md(
    """## Context & Methods

### Key Assumptions

- The full horizontal GR trace and typewell are legal inputs.
- Affine GR calibration uses only rows with visible `TVT_input`.
- Candidate TVT paths are centered on the previously cross-fitted Ridge prediction.
- The correction is exactly zero at projection start and reaches its offset after 15% of the suffix.
- Search is bounded to `±60 ft`; two modes must be separated by at least `9 ft`.
- Validation targets enter only scores and oracle diagnostics.
- Spatially stratified folds are reporting slices; the per-well aligner fits no cross-well model.

Promotion gate: primary RMSE below Ridge `14.749`, wins at least 4/5 slices, does not worsen worst-decile SSE share, with a stretch target of `≤13.0`.
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

VERSION = "v1"
SEED = 42
N_FOLDS = 5
STRIDE = 5
SHIFT_GRID = np.arange(-60.0, 60.01, 3.0)
SMOKE = os.environ.get("ROGII_SMOKE", "0") == "1"
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
LEGAL_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
PRIMARY = "denoise_r15_posterior_g25"


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


# Runnable checks: anchor stays fixed and the two modes are separated.
assert anchored_ramp([0.0, 0.15, 1.0], 0.15)[0] == 0.0
demo_cost = (SHIFT_GRID - 12.0) ** 2
demo_first, demo_second = separated_modes(demo_cost)
assert abs(SHIFT_GRID[demo_first] - SHIFT_GRID[demo_second]) >= 9.0
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
    prediction_frames.append(pd.DataFrame({"well_id": well_id, "target": truth, **predictions}))
    diagnostics.append(record)
    if number % 100 == 0:
        print("aligned", number, "/", len(ids))

OOF = pd.concat(prediction_frames, ignore_index=True)
WELLS = pd.DataFrame(diagnostics)
CANDIDATES = [column for column in OOF.columns if column not in {"well_id", "target"}]
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
ridge_tail = float(score_index.loc["ridge_prior", "worst10_sse_share"])
primary_tail = float(score_index.loc[PRIMARY, "worst10_sse_share"])
pivot = FOLD_SCORES.pivot(index="fold", columns="candidate", values="pooled_rmse")
fold_wins = int((pivot[PRIMARY] < pivot["ridge_prior"]).sum())
viable = bool(primary_score < ridge_score and fold_wins >= 4 and primary_tail <= ridge_tail)
stretch = bool(viable and primary_score <= 13.0)

display(SCORES.head(15))
display(pivot[["ridge_prior", PRIMARY, "oracle_config"]])
display(WELLS[["calibration_r2", "calibration_gain", "sigma", "denoise_r15_probability1", "denoise_r15_posterior_shift"]].describe())
print({"primary": PRIMARY, "improvement_ft": ridge_score - primary_score, "fold_wins": fold_wins, "viable": viable, "stretch": stretch})
'''
)

md("## Takeaways and artifacts")

code(
    r'''SCORES.to_csv(WORK / "heel_bimodal_scores_v1.csv", index=False)
FOLD_SCORES.to_csv(WORK / "heel_bimodal_fold_scores_v1.csv", index=False)
WELLS.to_csv(WORK / "heel_bimodal_wells_v1.csv", index=False)
OOF.to_parquet(WORK / "heel_bimodal_oof_v1.parquet", index=False)
summary = {
    "version": VERSION, "wells": int(WELLS.shape[0]), "rows": int(len(OOF)), "primary": PRIMARY,
    "ridge_rmse": ridge_score, "primary_rmse": primary_score, "improvement_ft": ridge_score - primary_score,
    "ridge_worst10_sse_share": ridge_tail, "primary_worst10_sse_share": primary_tail,
    "fold_wins": fold_wins, "viable": viable, "stretch_target_met": stretch,
    "scores": SCORES.head(15).to_dict("records"),
    "viability_rule": "primary beats Ridge, wins >=4/5 spatially stratified slices, and does not worsen worst10 SSE share",
    "decision": "build_piecewise_path" if viable else "revise_or_stop_heel_alignment",
    "caveats": [
        "Ridge center is imported from an earlier cross-fitted random-well experiment.",
        "Fixed variants share this OOF screen; only the predeclared primary controls promotion.",
        "Spatially stratified folds are evaluation slices because the aligner is per-well and fits no cross-well model.",
        "No submission is created.",
    ],
}
(WORK / "heel_bimodal_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
