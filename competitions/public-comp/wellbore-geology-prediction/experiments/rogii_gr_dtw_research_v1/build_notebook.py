import csv
import io
from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).with_name("rogii_gr_dtw_research_v1.ipynb")
RIDGE_SOURCE = OUT.parent.parent / "rogii_candidate_oracle_audit_v1" / "ranker_ridge_coefficients_v4.csv"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# ROGII multi-scale GR-DTW research v3

## tl;dr

This version tests whether the cross-fitted V2 GR correction can improve the exact public-anchor candidate on a same-ID pseudo-test.

No `submission.csv` is created. Positive residual doses are rejected unless they beat the frozen anchor on the three public well IDs.
"""
)

md(
    """## Context & Methods

Each suffix row receives a discrete offset state in `[-90, 90] ft` around the Ridge path. Emission cost compares horizontal and typewell GR at raw, rolling-21, and rolling-61 scales. Viterbi transitions bound local offset changes and penalize rough paths.

### Key Assumptions

- The Ridge center is cross-fitted by complete well and frozen from the geometry experiment.
- Validation TVT never enters emissions, path decoding, or configuration parameters.
- Typewell GR is affinely calibrated only on the visible `TVT_input` prefix.
- The stride-10 path is linearly interpolated back to every suffix row.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

VERSION = "v3"
SMOKE = os.environ.get("ROGII_SMOKE", "0") == "1"
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
LEGAL_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
STRIDE = 10
OFFSET_GRID = np.arange(-90.0, 90.1, 3.0)
CONFIGS = {
    "dtw_raw": {"emission": "raw", "transition": 0.5, "radius": 2, "center": 0.01, "gain": 0.2},
    "dtw_multi_soft": {"emission": "multi", "transition": 0.15, "radius": 3, "center": 0.002, "gain": 0.1},
    "dtw_multi_stiff": {"emission": "multi", "transition": 1.0, "radius": 2, "center": 0.01, "gain": 0.2},
}


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
all_ids = sorted(path.name.removesuffix(HORIZONTAL_SUFFIX) for path in TRAIN_DIR.glob(f"*{HORIZONTAL_SUFFIX}"))
PILOT_IDS = set(all_ids[:20])
ids = all_ids
if SMOKE:
    ids = ids[:20]


def rmse(y_true, prediction):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(prediction, float)) ** 2)))


def filled(values):
    series = pd.Series(values, dtype=float).interpolate(limit_direction="both")
    fallback = float(series.median()) if series.notna().any() else 0.0
    return series.fillna(fallback).to_numpy(float)


def smooth(values, window):
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy(float)


print({"version": VERSION, "root": str(ROOT), "wells": len(ids), "states": len(OFFSET_GRID), "smoke": SMOKE})
'''
)

md("## Data - legal Ridge center and calibrated GR emissions")

code(
    r'''def well_inputs(well_id):
    frame = pd.read_csv(TRAIN_DIR / f"{well_id}{HORIZONTAL_SUFFIX}", usecols=LEGAL_COLUMNS + ["TVT"])
    typewell = pd.read_csv(TRAIN_DIR / f"{well_id}{TYPEWELL_SUFFIX}", usecols=["TVT", "GR"])
    known = frame["TVT_input"].notna().to_numpy()
    known_idx, target_idx = np.flatnonzero(known), np.flatnonzero(~known)
    ps = known_idx[-1]
    md_values = frame["MD"].to_numpy(float)
    z = frame["Z"].to_numpy(float)
    tvt_input = frame["TVT_input"].to_numpy(float)
    span = max(float(md_values[target_idx[-1]] - md_values[ps]), 1.0)
    x = (md_values[target_idx] - md_values[ps]) / span
    base = float(tvt_input[ps]) - (z[target_idx] - z[ps])
    coefficient = ridge_coefficients.loc[well_id]
    ridge = base + float(coefficient["ridge_c1"]) * x + float(coefficient["ridge_c2"]) * x * x
    truth = frame["TVT"].to_numpy(float)[target_idx]

    horizontal_gr = filled(frame["GR"])
    typewell = typewell.dropna().sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = typewell["TVT"].to_numpy(float)
    tw_gr = filled(typewell["GR"])
    reference = np.interp(tvt_input[known_idx], tw_tvt, tw_gr)
    scale, bias = np.linalg.lstsq(np.c_[reference, np.ones(len(reference))], horizontal_gr[known_idx], rcond=None)[0]
    scale = float(np.clip(scale, 0.3, 3.0))
    calibrated = scale * tw_gr + float(bias)
    residual = horizontal_gr[known_idx] - np.interp(tvt_input[known_idx], tw_tvt, calibrated)
    sigma = float(np.clip(1.4826 * np.median(np.abs(residual - np.median(residual))), 8.0, 50.0))
    return frame, target_idx, truth, ridge, horizontal_gr, tw_tvt, calibrated, sigma


def emission_costs(target_idx, ridge, horizontal_gr, tw_tvt, tw_gr, sigma):
    sample_local = np.unique(np.r_[np.arange(0, len(target_idx), STRIDE), len(target_idx) - 1])
    sample_rows = target_idx[sample_local]
    candidates = ridge[sample_local, None] + OFFSET_GRID[None, :]
    costs = []
    for window in [1, 21, 61]:
        horizontal_scale = horizontal_gr if window == 1 else smooth(horizontal_gr, window)
        typewell_scale = tw_gr if window == 1 else smooth(tw_gr, window)
        expected = np.interp(candidates.ravel(), tw_tvt, typewell_scale).reshape(candidates.shape)
        residual = (horizontal_scale[sample_rows, None] - expected) / sigma
        costs.append(np.minimum(residual * residual, 16.0))
    outside = ((candidates < tw_tvt[0]) | (candidates > tw_tvt[-1])).astype(float) * 16.0
    return sample_local, costs[0] + outside, np.mean(costs, axis=0) + outside
'''
)

md("## Methods - slope-constrained Viterbi")

code(
    r'''def decode_offsets(emission, transition, radius, center_penalty):
    n_rows, n_states = emission.shape
    back = np.zeros((n_rows, n_states), dtype=np.int16)
    dp = emission[0] + 2.0 * (OFFSET_GRID / 6.0) ** 2
    center_cost = center_penalty * (OFFSET_GRID / 30.0) ** 2
    for row in range(1, n_rows):
        new = np.full(n_states, np.inf)
        source = np.zeros(n_states, dtype=np.int16)
        for shift in range(-radius, radius + 1):
            destination = np.arange(max(0, shift), min(n_states, n_states + shift))
            origin = destination - shift
            candidate = dp[origin] + transition * shift * shift
            better = candidate < new[destination]
            new[destination[better]] = candidate[better]
            source[destination[better]] = origin[better]
        dp = new + emission[row] + center_cost
        back[row] = source
    state = int(np.argmin(dp))
    states = np.empty(n_rows, dtype=int)
    states[-1] = state
    for row in range(n_rows - 1, 0, -1):
        state = int(back[row, state])
        states[row - 1] = state
    return OFFSET_GRID[states]


# Runnable self-check: a zero-emission path stays at the anchored zero offset.
zero_path = decode_offsets(np.zeros((8, len(OFFSET_GRID))), 1.0, 2, 0.01)
assert np.max(np.abs(zero_path)) == 0.0
'''
)

md("## Results - full-row OOF predictions")

code(
    r'''prediction_frames = []
diagnostic_rows = []
start_time = time.time()

for index, well_id in enumerate(ids, 1):
    frame, target_idx, truth, ridge, horizontal_gr, tw_tvt, tw_gr, sigma = well_inputs(well_id)
    sample_local, raw_emission, multi_emission = emission_costs(target_idx, ridge, horizontal_gr, tw_tvt, tw_gr, sigma)
    predictions = {"ridge_prior": ridge}
    offsets = {}
    for name, config in CONFIGS.items():
        emission = raw_emission if config["emission"] == "raw" else multi_emission
        sample_offsets = decode_offsets(emission, config["transition"], config["radius"], config["center"])
        full_offsets = np.interp(np.arange(len(target_idx)), sample_local, sample_offsets)
        predictions[name] = ridge + config["gain"] * full_offsets
        offsets[name] = full_offsets

    configuration_errors = {name: rmse(truth, predictions[name]) for name in CONFIGS}
    oracle_name = min(configuration_errors, key=configuration_errors.get)
    predictions["oracle_config"] = predictions[oracle_name]
    prediction_frames.append(pd.DataFrame({
        "id": [f"{well_id}_{row}" for row in target_idx], "well_id": well_id,
        "row_index": target_idx, "target": truth, **predictions,
    }))
    diagnostic_rows.append({
        "well_id": well_id, "rows": len(target_idx), "robust_sigma": sigma,
        "oracle_config": oracle_name,
        **{f"{name}_rmse": configuration_errors[name] for name in CONFIGS},
        **{f"{name}_offset_end": float(value[-1]) for name, value in offsets.items()},
        **{f"{name}_boundary_share": float((np.abs(value) >= OFFSET_GRID.max()).mean()) for name, value in offsets.items()},
    })
    if index % 100 == 0:
        print("decoded", index, "/", len(ids), "elapsed", round(time.time() - start_time, 1))

OOF = pd.concat(prediction_frames, ignore_index=True)
WELL_DIAGNOSTICS = pd.DataFrame(diagnostic_rows)
BASE_CANDIDATES = ["ridge_prior", *CONFIGS, "oracle_config"]
assert OOF["id"].is_unique and np.isfinite(OOF[BASE_CANDIDATES].to_numpy(float)).all()
print("OOF", OOF.shape)
'''
)

md("## Methods - nested confidence gating")

code(
    r'''diagnostics_by_well = WELL_DIAGNOSTICS.set_index("well_id")
meta_rows, optimal_raw_gain = [], {}
for well_id, group in OOF.groupby("well_id", sort=True):
    ridge = group["ridge_prior"].to_numpy(float)
    residual = group["target"].to_numpy(float) - ridge
    row = {"well_id": well_id, "rows": len(group), "robust_sigma": diagnostics_by_well.loc[well_id, "robust_sigma"]}
    for name, config in CONFIGS.items():
        offset = (group[name].to_numpy(float) - ridge) / config["gain"]
        row.update({
            f"{name}_mean": offset.mean(), f"{name}_std": offset.std(),
            f"{name}_abs_mean": np.abs(offset).mean(),
            f"{name}_roughness": np.abs(np.diff(offset)).mean(),
            f"{name}_end": offset[-1],
            f"{name}_zero_share": (np.abs(offset) < 1e-9).mean(),
            f"{name}_boundary_share": (np.abs(offset) >= OFFSET_GRID.max()).mean(),
        })
        if name == "dtw_raw":
            denominator = offset @ offset
            optimal_raw_gain[well_id] = (residual @ offset) / denominator if denominator else 0.0
    meta_rows.append(row)

META = pd.DataFrame(meta_rows).set_index("well_id")
raw_offset = (OOF["dtw_raw"].to_numpy(float) - OOF["ridge_prior"].to_numpy(float)) / CONFIGS["dtw_raw"]["gain"]
nested_global = np.empty(len(OOF))
nested_gate = np.empty(len(OOF))
global_gain_by_well, gate_gain_by_well = {}, {}

for train_index, valid_index in KFold(5, shuffle=True, random_state=42).split(META):
    train_ids, valid_ids = META.index[train_index], META.index[valid_index]
    train_mask = OOF["well_id"].isin(train_ids).to_numpy()
    valid_mask = OOF["well_id"].isin(valid_ids).to_numpy()
    global_gain = float(
        ((OOF.loc[train_mask, "target"] - OOF.loc[train_mask, "ridge_prior"]) * raw_offset[train_mask]).sum()
        / (raw_offset[train_mask] @ raw_offset[train_mask])
    )
    model = ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=15, max_features=0.8, random_state=42, n_jobs=-1,
    ).fit(
        META.iloc[train_index], pd.Series(optimal_raw_gain).loc[train_ids],
        sample_weight=META.iloc[train_index]["rows"],
    )
    gate_map = pd.Series(np.clip(model.predict(META.iloc[valid_index]), -0.05, 0.2), index=valid_ids)
    valid_gain = OOF.loc[valid_mask, "well_id"].map(gate_map).to_numpy(float)
    ridge_valid = OOF.loc[valid_mask, "ridge_prior"].to_numpy(float)
    nested_global[valid_mask] = ridge_valid + global_gain * raw_offset[valid_mask]
    nested_gate[valid_mask] = ridge_valid + valid_gain * raw_offset[valid_mask]
    global_gain_by_well.update(dict.fromkeys(valid_ids, global_gain))
    gate_gain_by_well.update(gate_map.to_dict())

OOF["nested_global_raw"] = nested_global
OOF["nested_gate_raw"] = nested_gate
WELL_DIAGNOSTICS["nested_global_gain"] = WELL_DIAGNOSTICS["well_id"].map(global_gain_by_well)
WELL_DIAGNOSTICS["nested_gate_gain"] = WELL_DIAGNOSTICS["well_id"].map(gate_gain_by_well)
CANDIDATES = ["ridge_prior", *CONFIGS, "nested_global_raw", "nested_gate_raw", "oracle_config"]
assert np.isfinite(OOF[CANDIDATES].to_numpy(float)).all()
assert WELL_DIAGNOSTICS["nested_gate_gain"].between(-0.05, 0.2).all()
'''
)

md("## Results - decision metrics")

code(
    r'''pilot_mask = OOF["well_id"].isin(PILOT_IDS)
confirmatory_mask = ~pilot_mask
SCORES = pd.DataFrame([{
    "candidate": candidate,
    "pooled_rmse": rmse(OOF["target"], OOF[candidate]),
    "pilot_rmse": rmse(OOF.loc[pilot_mask, "target"], OOF.loc[pilot_mask, candidate]),
    "confirmatory_rmse": (
        rmse(OOF.loc[confirmatory_mask, "target"], OOF.loc[confirmatory_mask, candidate])
        if confirmatory_mask.any() else np.nan
    ),
} for candidate in CANDIDATES])
selection_metric = "confirmatory_rmse" if confirmatory_mask.any() else "pooled_rmse"
SCORES = SCORES.sort_values(selection_metric).reset_index(drop=True)

for candidate in CANDIDATES:
    per_well = OOF.groupby("well_id", sort=False).apply(
        lambda group: rmse(group["target"], group[candidate]), include_groups=False
    )
    WELL_DIAGNOSTICS[f"{candidate}_full_rmse"] = WELL_DIAGNOSTICS["well_id"].map(per_well)

display(SCORES)
display(WELL_DIAGNOSTICS[["robust_sigma", *[f"{name}_boundary_share" for name in CONFIGS]]].describe())
display(WELL_DIAGNOSTICS["oracle_config"].value_counts())
'''
)

md("## Results - public-ID pseudo-test against the 6.979 anchor")

code(
    r'''import hashlib

ANCHOR_SHA256 = "0c60510dc11f7750c493c29c75ac9383eb6ea331d976a0c7991a895c700e7cf8"
anchor_candidates = []
if os.environ.get("ROGII_ANCHOR_SUBMISSION"):
    anchor_candidates.append(Path(os.environ["ROGII_ANCHOR_SUBMISSION"]))
if Path("/kaggle/input").exists():
    anchor_candidates.extend(Path("/kaggle/input").rglob("submission_model_package_gated_015.csv"))
anchor_path = next((path for path in anchor_candidates if path.exists()), None)
if anchor_path is None:
    raise FileNotFoundError("Frozen 6.979 anchor candidate not found")
anchor_sha256 = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
assert anchor_sha256 == ANCHOR_SHA256, f"Unexpected anchor hash: {anchor_sha256}"

public_ids = []
for path in sorted((ROOT / "test").glob(f"*{HORIZONTAL_SUFFIX}")):
    well_id = path.name.removesuffix(HORIZONTAL_SUFFIX)
    test_frame = pd.read_csv(path, usecols=["TVT_input"])
    public_ids.extend(f"{well_id}_{row}" for row in np.flatnonzero(test_frame["TVT_input"].isna()))

anchor = pd.read_csv(anchor_path, usecols=["id", "tvt"])
anchor["id"] = anchor["id"].astype(str)
anchor = anchor[anchor["id"].isin(public_ids)]
PSEUDO = anchor.merge(
    OOF[["id", "well_id", "row_index", "target", "ridge_prior", "nested_gate_raw"]],
    on="id", how="inner", validate="one_to_one",
)
assert set(PSEUDO["id"]) == set(public_ids) and len(PSEUDO) == len(public_ids)

correction = PSEUDO["nested_gate_raw"] - PSEUDO["ridge_prior"]
PSEUDO["anchor_6979"] = PSEUDO["tvt"]
for alpha in [0.1, 0.25, 1.0]:
    PSEUDO[f"anchor_plus_gate_{int(alpha * 100):03d}"] = PSEUDO["tvt"] + alpha * correction
PSEUDO_CANDIDATES = ["anchor_6979", "anchor_plus_gate_010", "anchor_plus_gate_025", "anchor_plus_gate_100"]
PSEUDO_SCORES = pd.DataFrame([
    {"candidate": name, "pooled_rmse": rmse(PSEUDO["target"], PSEUDO[name])}
    for name in PSEUDO_CANDIDATES
]).sort_values("pooled_rmse").reset_index(drop=True)

pseudo_well_rows = []
for well_id, group in PSEUDO.groupby("well_id", sort=True):
    local_correction = group["nested_gate_raw"].to_numpy() - group["ridge_prior"].to_numpy()
    denominator = local_correction @ local_correction
    pseudo_well_rows.append({
        "well_id": well_id, "rows": len(group),
        **{f"{name}_rmse": rmse(group["target"], group[name]) for name in PSEUDO_CANDIDATES},
        "gate_correction_mean": float(local_correction.mean()),
        "gate_correction_rms": float(np.sqrt(np.mean(local_correction ** 2))),
        "oracle_alpha": float(((group["target"].to_numpy() - group["tvt"].to_numpy()) @ local_correction) / denominator) if denominator else 0.0,
    })
PSEUDO_WELLS = pd.DataFrame(pseudo_well_rows)
anchor_well_score = PSEUDO_WELLS.set_index("well_id")["anchor_6979_rmse"]
eligible = []
for candidate in PSEUDO_CANDIDATES[1:]:
    score = float(PSEUDO_SCORES.set_index("candidate").loc[candidate, "pooled_rmse"])
    per_well = PSEUDO_WELLS.set_index("well_id")[f"{candidate}_rmse"]
    if score < float(PSEUDO_SCORES.set_index("candidate").loc["anchor_6979", "pooled_rmse"]) and (per_well < anchor_well_score).all():
        eligible.append(candidate)

display(PSEUDO_SCORES)
display(PSEUDO_WELLS)
print({"anchor": str(anchor_path), "sha256": anchor_sha256, "submission_eligible": eligible})
'''
)

md("## Takeaways and artifacts")

code(
    r'''SCORES.to_csv(WORK / "gr_dtw_scores_v3.csv", index=False)
WELL_DIAGNOSTICS.to_csv(WORK / "gr_dtw_well_diagnostics_v3.csv", index=False)
PSEUDO_SCORES.to_csv(WORK / "gr_dtw_public_pseudo_scores_v3.csv", index=False)
PSEUDO_WELLS.to_csv(WORK / "gr_dtw_public_pseudo_wells_v3.csv", index=False)
prediction_path = WORK / "gr_dtw_oof_predictions_v3.parquet"
OOF.to_parquet(prediction_path, index=False)

deployable = SCORES[SCORES["candidate"] != "oracle_config"].iloc[0]
score_records = SCORES.astype(object).where(SCORES.notna(), None).to_dict("records")
summary = {
    "version": VERSION, "wells": int(OOF["well_id"].nunique()), "rows": int(len(OOF)),
    "stride": STRIDE, "offset_grid": [float(OFFSET_GRID.min()), float(OFFSET_GRID.max()), 3.0],
    "configs": CONFIGS, "scores": score_records,
    "selection_metric": selection_metric,
    "best_deployable": str(deployable["candidate"]),
    "best_deployable_rmse": float(deployable[selection_metric]),
    "ridge_prior_rmse": float(SCORES.set_index("candidate").loc["ridge_prior", selection_metric]),
    "public_anchor_expected_lb": 6.979,
    "public_anchor_sha256": anchor_sha256,
    "public_pseudo_scores": PSEUDO_SCORES.to_dict("records"),
    "submission_eligible": eligible,
    "submission_decision": "build_candidate" if eligible else "reject_gate_on_anchor",
    "prediction_artifact": prediction_path.name,
    "caveats": [
        "Nested gain predictions are cross-fitted by complete well; held-out TVT never enters its gain.",
        "Fixed configurations are screened on the same OOF and are not yet nested-selected.",
        "Gains were selected on the first 20 sorted wells; confirmatory_rmse excludes those pilot wells.",
        "The Ridge center is imported from a prior cross-fitted experiment.",
        "The public-ID pseudo-test uses train TVT as a proxy; hidden leaderboard TVT is unavailable.",
        "No submission or test-set correction is produced.",
    ],
}
(WORK / "gr_dtw_summary_v3.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
