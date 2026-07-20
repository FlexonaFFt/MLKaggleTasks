import csv
import io
from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).with_name("rogii_neighbor_transfer_research_v1.ipynb")
RIDGE_SOURCE = OUT.parent.parent / "rogii_candidate_oracle_audit_v1" / "ranker_ridge_coefficients_v4.csv"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# ROGII neighbor-transfer research v3

## tl;dr

V2 produced a nested OOF score of 13.480 versus the 14.749 Ridge fallback. V3 audits whether that held-out-well experiment can produce new leaderboard information on the actual three-well test set.

It never creates `submission.csv`. The new decision artifact is `neighbor_test_audit_v3.csv`; V2 OOF artifacts are retained with V3 names.
"""
)

md(
    """## Context & Methods

The old neighbor proxy copied absolute TVT by MD and scored poorly. This version transfers only the residual shape around the validated geometric family:

`correction = TVT - [TVT_PS - (Z - Z_PS)]`

### Key Assumptions

- Validation suffix targets never participate in neighbor selection.
- Outer validation wells use only outer-training references; inner calibration also excludes the outer fold and its own inner fold.
- Full `X/Y/Z/MD` trajectories and drilling direction are legal at inference.
- The prior Ridge coefficients come from the already completed cross-fitted geometry notebook.
- Coordinate (`MD-after-PS` or normalized suffix) and threshold are selected from a fixed grid using pooled inner OOF SSE.
- The test audit checks exact train-ID overlap before authorizing any leaderboard submission.
- Random well folds model the case where evaluation wells have nearby interpreted training wells; spatial holdout remains a later robustness test.
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

warnings.filterwarnings("ignore")

VERSION = "v3"
SEED = 42
N_SPLITS = 5
SMOKE = os.environ.get("ROGII_SMOKE", "0") == "1"
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
LEGAL_COLUMNS = ["MD", "X", "Y", "Z", "TVT_input"]


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
TEST_DIR = ROOT / "test"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()


def rmse(y_true, prediction):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(prediction, float)) ** 2)))


all_ids = sorted(path.name.removesuffix(HORIZONTAL_SUFFIX) for path in TRAIN_DIR.glob(f"*{HORIZONTAL_SUFFIX}"))
ids = all_ids[:20] if SMOKE else all_ids
if SMOKE:
    N_SPLITS = 3

rng = np.random.default_rng(SEED)
shuffled = np.array(ids, dtype=object)
rng.shuffle(shuffled)
FOLD_BY_WELL = {well_id: int(index % N_SPLITS) for index, well_id in enumerate(shuffled)}

ridge_coefficients = pd.read_csv(StringIO(RIDGE_CSV)).set_index("well_id", drop=False)
missing_coefficients = sorted(set(ids) - set(ridge_coefficients.index))
assert not missing_coefficients, missing_coefficients[:5]
print({"version": VERSION, "root": str(ROOT), "wells": len(ids), "folds": N_SPLITS, "smoke": SMOKE, "ridge_rows": len(ridge_coefficients)})
'''
)

md("## Data - build legal trajectory profiles")

code(
    r'''def build_profile(well_id):
    frame = pd.read_csv(TRAIN_DIR / f"{well_id}{HORIZONTAL_SUFFIX}", usecols=LEGAL_COLUMNS + ["TVT"])
    known = frame["TVT_input"].notna().to_numpy()
    known_idx = np.flatnonzero(known)
    target_idx = np.flatnonzero(~known)
    if len(known_idx) < 20 or len(target_idx) < 20:
        return None

    ps = known_idx[-1]
    md_values = frame["MD"].to_numpy(float)
    x_coord = frame["X"].to_numpy(float)
    y_coord = frame["Y"].to_numpy(float)
    z = frame["Z"].to_numpy(float)
    tvt_input = frame["TVT_input"].to_numpy(float)
    truth = frame["TVT"].to_numpy(float)[target_idx]
    suffix_span = max(float(md_values[target_idx[-1]] - md_values[ps]), 1.0)
    md_after_ps = md_values[target_idx] - md_values[ps]
    normalized_md = md_after_ps / suffix_span
    z_anchor = float(tvt_input[ps]) - (z[target_idx] - z[ps])
    correction = truth - z_anchor
    azimuth = float(np.arctan2(y_coord[target_idx[-1]] - y_coord[ps], x_coord[target_idx[-1]] - x_coord[ps]))

    path_grid = np.linspace(0.0, 1.0, 25)
    path_x = np.interp(path_grid, np.r_[0.0, normalized_md], np.r_[x_coord[ps], x_coord[target_idx]])
    path_y = np.interp(path_grid, np.r_[0.0, normalized_md], np.r_[y_coord[ps], y_coord[target_idx]])
    return {
        "well_id": well_id, "fold": FOLD_BY_WELL[well_id], "target_idx": target_idx,
        "truth": truth, "normalized_md": normalized_md, "md_after_ps": md_after_ps,
        "z_anchor": z_anchor, "last_known": float(tvt_input[ps]), "correction": correction,
        "ps_x": float(x_coord[ps]), "ps_y": float(y_coord[ps]), "azimuth": azimuth,
        "direction": 1 if np.cos(azimuth) >= 0 else -1,
        "path_xy": np.column_stack([path_x, path_y]),
    }


PROFILES = {}
for index, well_id in enumerate(ids, 1):
    profile = build_profile(well_id)
    if profile is not None:
        PROFILES[well_id] = profile
    if index % 100 == 0:
        print("profiled", index, "/", len(ids))

ids = sorted(PROFILES)
assert len(ids) >= 10 and all(np.isfinite(PROFILES[well_id]["truth"]).all() for well_id in ids)
print("usable wells", len(ids), "suffix rows", sum(len(PROFILES[well_id]["truth"]) for well_id in ids))
'''
)

md("## Methods - spatial neighbor and residual transfer")

code(
    r'''def angular_difference(left, right):
    return float(np.arccos(np.clip(np.cos(left - right), -1.0, 1.0)))


def ranked_neighbors(well_id, reference_ids, metric):
    query = PROFILES[well_id]
    same_direction = [ref for ref in reference_ids if PROFILES[ref]["direction"] == query["direction"]]
    candidates = same_direction or list(reference_ids)
    if metric == "ps":
        distances = np.array([
            np.hypot(PROFILES[ref]["ps_x"] - query["ps_x"], PROFILES[ref]["ps_y"] - query["ps_y"])
            for ref in candidates
        ])
    elif metric == "path":
        reference_paths = np.stack([PROFILES[ref]["path_xy"] for ref in candidates])
        distances = np.median(np.linalg.norm(reference_paths - query["path_xy"][None, :, :], axis=2), axis=1)
    else:
        raise ValueError(metric)
    order = np.argsort(distances)
    return [(candidates[index], float(distances[index])) for index in order]


def transferred_correction(target_id, reference_id, coordinate):
    target = PROFILES[target_id]
    reference = PROFILES[reference_id]
    if coordinate == "normalized":
        query_axis = target["normalized_md"]
        reference_axis = reference["normalized_md"]
    elif coordinate == "md":
        query_axis = target["md_after_ps"]
        reference_axis = reference["md_after_ps"]
    else:
        raise ValueError(coordinate)
    correction = np.interp(
        query_axis,
        np.r_[0.0, reference_axis],
        np.r_[0.0, reference["correction"]],
        left=0.0,
        right=float(reference["correction"][-1]),
    )
    return target["z_anchor"] + correction


def ridge_prediction(well_id):
    profile = PROFILES[well_id]
    coefficient = ridge_coefficients.loc[well_id]
    x = profile["normalized_md"]
    return profile["z_anchor"] + float(coefficient["ridge_c1"]) * x + float(coefficient["ridge_c2"]) * x * x


# Small self-check: every transferred curve stays anchored at zero correction at PS.
first_id, second_id = ids[:2]
assert abs(np.interp(0.0, np.r_[0.0, PROFILES[second_id]["normalized_md"]], np.r_[0.0, PROFILES[second_id]["correction"]])) < 1e-12
'''
)

md("## Results - nested calibration and outer OOF")

code(
    r'''COORDINATES = ["normalized", "md"]
THRESHOLDS = [150, 300, 450, 600, 750, 900, 1200]


def candidate_bundle(well_id, reference_ids):
    profile = PROFILES[well_id]
    path_ranked = ranked_neighbors(well_id, reference_ids, "path")
    neighbor_id, distance = path_ranked[0]
    assert neighbor_id in reference_ids
    predictions = {
        coordinate: transferred_correction(well_id, neighbor_id, coordinate)
        for coordinate in COORDINATES
    }
    return path_ranked, neighbor_id, distance, predictions


prediction_frames = []
well_rows = []
calibration_rows = []
start_time = time.time()

for outer_fold in range(N_SPLITS):
    outer_valid = [well_id for well_id in ids if FOLD_BY_WELL[well_id] == outer_fold]
    outer_train = [well_id for well_id in ids if FOLD_BY_WELL[well_id] != outer_fold]
    tuning_rows = []

    for well_id in outer_train:
        inner_references = [
            ref for ref in outer_train if FOLD_BY_WELL[ref] != FOLD_BY_WELL[well_id]
        ]
        assert inner_references and all(FOLD_BY_WELL[ref] != outer_fold for ref in inner_references)
        _, _, distance, predictions = candidate_bundle(well_id, inner_references)
        truth = PROFILES[well_id]["truth"]
        tuning_rows.append({
            "well_id": well_id, "rows": len(truth), "distance": distance,
            "ridge_sse": float(np.sum((truth - ridge_prediction(well_id)) ** 2)),
            "normalized_sse": float(np.sum((truth - predictions["normalized"]) ** 2)),
            "md_sse": float(np.sum((truth - predictions["md"]) ** 2)),
        })

    tuning = pd.DataFrame(tuning_rows)
    settings = []
    for coordinate in COORDINATES:
        for threshold in THRESHOLDS:
            selected_sse = np.where(
                tuning["distance"] <= threshold,
                tuning[f"{coordinate}_sse"], tuning["ridge_sse"],
            )
            settings.append({
                "coordinate": coordinate, "threshold": threshold,
                "inner_rmse": float(np.sqrt(selected_sse.sum() / tuning["rows"].sum())),
            })
    best = min(settings, key=lambda row: row["inner_rmse"])
    calibration_rows.append({"fold": outer_fold, "tuning_wells": len(tuning), **best})

    for well_id in outer_valid:
        profile = PROFILES[well_id]
        path_ranked, neighbor_id, distance, predictions = candidate_bundle(well_id, outer_train)
        oracle_predictions = [
            transferred_correction(well_id, ref, coordinate)
            for ref, _ in path_ranked[:5] for coordinate in COORDINATES
        ]
        oracle_errors = [rmse(profile["truth"], prediction) for prediction in oracle_predictions]
        oracle_position = int(np.argmin(oracle_errors))
        top5x2_oracle = oracle_predictions[oracle_position]
        ridge = ridge_prediction(well_id)
        nested = predictions[best["coordinate"]] if distance <= best["threshold"] else ridge

        prediction_frames.append(pd.DataFrame({
            "id": [f"{well_id}_{row}" for row in profile["target_idx"]],
            "well_id": well_id, "fold": outer_fold, "target": profile["truth"],
            "last_known": profile["last_known"], "z_anchor": profile["z_anchor"],
            "ridge_prior": ridge,
            "neighbor_path_normalized": predictions["normalized"],
            "neighbor_path_md": predictions["md"],
            "hybrid_normalized_600": predictions["normalized"] if distance <= 600 else ridge,
            "hybrid_md_600": predictions["md"] if distance <= 600 else ridge,
            "nested_selector": nested, "top5x2_oracle": top5x2_oracle,
        }))
        well_rows.append({
            "well_id": well_id, "fold": outer_fold, "rows": len(profile["truth"]),
            "direction": profile["direction"], "path_neighbor": neighbor_id,
            "path_distance": distance,
            "path_angle_difference_deg": np.degrees(angular_difference(profile["azimuth"], PROFILES[neighbor_id]["azimuth"])),
            "selected_coordinate": best["coordinate"], "selected_threshold": best["threshold"],
            "top5x2_oracle_choice": oracle_position + 1,
        })
    print("outer fold", outer_fold, "choice", best, "elapsed", round(time.time() - start_time, 1))

OOF = pd.concat(prediction_frames, ignore_index=True)
WELL_DIAGNOSTICS = pd.DataFrame(well_rows)
CALIBRATION = pd.DataFrame(calibration_rows)
CANDIDATES = [
    "last_known", "z_anchor", "ridge_prior",
    "neighbor_path_normalized", "neighbor_path_md",
    "hybrid_normalized_600", "hybrid_md_600", "nested_selector", "top5x2_oracle",
]
assert OOF["id"].is_unique and np.isfinite(OOF[CANDIDATES].to_numpy(float)).all()
assert WELL_DIAGNOSTICS["well_id"].nunique() == len(ids)
print("OOF", OOF.shape)
display(CALIBRATION)
'''
)

md("## Results - pooled scores and distance regimes")

code(
    r'''SCORES = pd.DataFrame([
    {"candidate": candidate, "pooled_rmse": rmse(OOF["target"], OOF[candidate])}
    for candidate in CANDIDATES
]).sort_values("pooled_rmse").reset_index(drop=True)

well_scores = []
for well_id, group in OOF.groupby("well_id", sort=False):
    row = {"well_id": well_id}
    for candidate in CANDIDATES:
        row[f"{candidate}_rmse"] = rmse(group["target"], group[candidate])
    well_scores.append(row)
WELL_DIAGNOSTICS = WELL_DIAGNOSTICS.merge(pd.DataFrame(well_scores), on="well_id", validate="one_to_one")

bins = [-np.inf, 150, 300, 600, np.inf]
labels = ["<150", "150-300", "300-600", ">600"]
WELL_DIAGNOSTICS["distance_bin"] = pd.cut(WELL_DIAGNOSTICS["path_distance"], bins=bins, labels=labels)
distance_by_well = WELL_DIAGNOSTICS.set_index("well_id")["distance_bin"]
OOF["distance_bin"] = OOF["well_id"].map(distance_by_well)

distance_rows = []
for distance_bin, group in OOF.groupby("distance_bin", observed=True):
    for candidate in ["ridge_prior", "neighbor_path_normalized", "neighbor_path_md", "nested_selector", "top5x2_oracle"]:
        distance_rows.append({
            "distance_bin": str(distance_bin), "candidate": candidate,
            "wells": int(group["well_id"].nunique()), "rows": len(group),
            "pooled_rmse": rmse(group["target"], group[candidate]),
        })
DISTANCE_SCORES = pd.DataFrame(distance_rows)

display(SCORES)
display(DISTANCE_SCORES.pivot(index="distance_bin", columns="candidate", values="pooled_rmse"))
display(WELL_DIAGNOSTICS[["path_distance", "path_angle_difference_deg", "top5x2_oracle_choice"]].describe())
'''
)

md("## Test-distribution audit")

code(
    r'''def test_geometry(well_id):
    frame = pd.read_csv(TEST_DIR / f"{well_id}{HORIZONTAL_SUFFIX}", usecols=LEGAL_COLUMNS)
    known_idx = np.flatnonzero(frame["TVT_input"].notna().to_numpy())
    target_idx = np.flatnonzero(frame["TVT_input"].isna().to_numpy())
    ps = known_idx[-1]
    md_values = frame["MD"].to_numpy(float)
    x_coord = frame["X"].to_numpy(float)
    y_coord = frame["Y"].to_numpy(float)
    span = max(float(md_values[target_idx[-1]] - md_values[ps]), 1.0)
    normalized_md = (md_values[target_idx] - md_values[ps]) / span
    grid = np.linspace(0.0, 1.0, 25)
    path_xy = np.column_stack([
        np.interp(grid, np.r_[0.0, normalized_md], np.r_[x_coord[ps], x_coord[target_idx]]),
        np.interp(grid, np.r_[0.0, normalized_md], np.r_[y_coord[ps], y_coord[target_idx]]),
    ])
    azimuth = float(np.arctan2(y_coord[target_idx[-1]] - y_coord[ps], x_coord[target_idx[-1]] - x_coord[ps]))
    return target_idx, path_xy, azimuth, 1 if np.cos(azimuth) >= 0 else -1


test_ids = sorted(path.name.removesuffix(HORIZONTAL_SUFFIX) for path in TEST_DIR.glob(f"*{HORIZONTAL_SUFFIX}"))
deployment_threshold = int(CALIBRATION["threshold"].median())
test_rows = []
for well_id in test_ids:
    target_idx, path_xy, azimuth, direction = test_geometry(well_id)
    same_id_present = well_id in PROFILES
    self_distance = float(np.median(np.linalg.norm(PROFILES[well_id]["path_xy"] - path_xy, axis=1))) if same_id_present else np.nan
    external = [ref for ref in ids if ref != well_id and PROFILES[ref]["direction"] == direction]
    paths = np.stack([PROFILES[ref]["path_xy"] for ref in external])
    distances = np.median(np.linalg.norm(paths - path_xy[None, :, :], axis=2), axis=1)
    nearest_index = int(np.argmin(distances))
    test_rows.append({
        "well_id": well_id, "prediction_rows": len(target_idx),
        "same_id_train_copy": same_id_present, "self_path_distance": self_distance,
        "external_neighbor": external[nearest_index], "external_path_distance": float(distances[nearest_index]),
        "external_neighbor_eligible": bool(distances[nearest_index] <= deployment_threshold),
        "deployment_threshold": deployment_threshold,
    })

TEST_AUDIT = pd.DataFrame(test_rows)
sample = pd.read_csv(ROOT / "sample_submission.csv")
assert TEST_AUDIT["prediction_rows"].sum() == len(sample)
assert TEST_AUDIT["well_id"].nunique() == len(test_ids)
display(TEST_AUDIT)
'''
)

md("## Takeaways and artifacts")

code(
    r'''SCORES.to_csv(WORK / "neighbor_scores_v3.csv", index=False)
DISTANCE_SCORES.to_csv(WORK / "neighbor_distance_scores_v3.csv", index=False)
WELL_DIAGNOSTICS.to_csv(WORK / "neighbor_well_diagnostics_v3.csv", index=False)
CALIBRATION.to_csv(WORK / "neighbor_calibration_v3.csv", index=False)
TEST_AUDIT.to_csv(WORK / "neighbor_test_audit_v3.csv", index=False)

prediction_path = WORK / "neighbor_oof_predictions_v3.parquet"
OOF.to_parquet(prediction_path, index=False)

best_deployable = SCORES[SCORES["candidate"] != "top5x2_oracle"].iloc[0]
summary = {
    "version": VERSION, "wells": int(OOF["well_id"].nunique()), "rows": int(len(OOF)),
    "scores": SCORES.to_dict("records"),
    "best_deployable": str(best_deployable["candidate"]),
    "best_deployable_rmse": float(best_deployable["pooled_rmse"]),
    "ridge_prior_rmse": float(SCORES.set_index("candidate").loc["ridge_prior", "pooled_rmse"]),
    "distance_coverage": {
        label: int((WELL_DIAGNOSTICS["distance_bin"].astype(str) == label).sum()) for label in labels
    },
    "median_path_distance": float(WELL_DIAGNOSTICS["path_distance"].median()),
    "calibration": CALIBRATION.to_dict("records"),
    "nested_selector_rmse": float(SCORES.set_index("candidate").loc["nested_selector", "pooled_rmse"]),
    "test_audit": TEST_AUDIT.to_dict("records"),
    "submission_decision": "skip_duplicate" if TEST_AUDIT["same_id_train_copy"].all() else "build_controlled_submission",
    "prediction_artifact": prediction_path.name,
    "caveats": [
        "Random well folds test access to nearby interpreted references; spatial holdout is not yet run.",
        "Top-5 x coordinate oracle uses validation targets and is diagnostic only.",
        "The fixed Ridge fallback is imported from a prior cross-fitted experiment rather than retrained inside each nested outer fold.",
        "No GR alignment, formation anchor, learned gate, or submission is included.",
    ],
}
(WORK / "neighbor_summary_v3.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
