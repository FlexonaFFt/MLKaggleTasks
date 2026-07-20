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
    """# ROGII neighbor-transfer research v1

## tl;dr

This is a diagnostic OOF notebook for one question: can the full TVT correction shape of a spatially close, same-direction training well improve a held-out well?

It never creates `submission.csv`. The executed decision artifacts are `neighbor_scores_v1.csv`, `neighbor_distance_scores_v1.csv`, `neighbor_well_diagnostics_v1.csv`, and `neighbor_summary_v1.json`.
"""
)

md(
    """## Context & Methods

The old neighbor proxy copied absolute TVT by MD and scored poorly. This version transfers only the residual shape around the validated geometric family:

`correction = TVT - [TVT_PS - (Z - Z_PS)]`

### Key Assumptions

- Validation suffix targets never participate in neighbor selection.
- Every validation well can use only reference wells from other folds.
- Full `X/Y/Z/MD` trajectories and drilling direction are legal at inference.
- The prior Ridge coefficients come from the already completed cross-fitted geometry notebook.
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

VERSION = "v1"
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
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()


def rmse(y_true, prediction):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(prediction, float)) ** 2)))


all_ids = sorted(path.name.removesuffix(HORIZONTAL_SUFFIX) for path in TRAIN_DIR.glob(f"*{HORIZONTAL_SUFFIX}"))
ids = all_ids[:20] if SMOKE else all_ids
if SMOKE:
    N_SPLITS = 2

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

md("## Results - well-grouped OOF")

code(
    r'''prediction_frames = []
well_rows = []
start_time = time.time()

for index, well_id in enumerate(ids, 1):
    profile = PROFILES[well_id]
    references = [ref for ref in ids if FOLD_BY_WELL[ref] != FOLD_BY_WELL[well_id]]
    ps_ranked = ranked_neighbors(well_id, references, "ps")
    path_ranked = ranked_neighbors(well_id, references, "path")
    ps_neighbor, ps_distance = ps_ranked[0]
    path_neighbor, path_distance = path_ranked[0]
    assert FOLD_BY_WELL[ps_neighbor] != FOLD_BY_WELL[well_id]
    assert FOLD_BY_WELL[path_neighbor] != FOLD_BY_WELL[well_id]

    neighbor_ps_normalized = transferred_correction(well_id, ps_neighbor, "normalized")
    neighbor_path_normalized = transferred_correction(well_id, path_neighbor, "normalized")
    neighbor_path_md = transferred_correction(well_id, path_neighbor, "md")

    top5_predictions = [transferred_correction(well_id, ref, "normalized") for ref, _ in path_ranked[:5]]
    top5_errors = [rmse(profile["truth"], prediction) for prediction in top5_predictions]
    oracle_position = int(np.argmin(top5_errors))
    top5_oracle = top5_predictions[oracle_position]
    oracle_neighbor = path_ranked[oracle_position][0]

    prediction_frames.append(pd.DataFrame({
        "id": [f"{well_id}_{row}" for row in profile["target_idx"]],
        "well_id": well_id, "fold": profile["fold"], "target": profile["truth"],
        "last_known": profile["last_known"], "z_anchor": profile["z_anchor"],
        "ridge_prior": ridge_prediction(well_id),
        "neighbor_ps_normalized": neighbor_ps_normalized,
        "neighbor_path_normalized": neighbor_path_normalized,
        "neighbor_path_md": neighbor_path_md,
        "top5_oracle": top5_oracle,
    }))
    well_rows.append({
        "well_id": well_id, "fold": profile["fold"], "rows": len(profile["truth"]),
        "direction": profile["direction"], "ps_neighbor": ps_neighbor, "ps_distance": ps_distance,
        "path_neighbor": path_neighbor, "path_distance": path_distance,
        "path_angle_difference_deg": np.degrees(angular_difference(profile["azimuth"], PROFILES[path_neighbor]["azimuth"])),
        "top5_oracle_neighbor": oracle_neighbor, "top5_oracle_rank": oracle_position + 1,
    })
    if index % 100 == 0:
        print("transferred", index, "/", len(ids), "elapsed", round(time.time() - start_time, 1))

OOF = pd.concat(prediction_frames, ignore_index=True)
WELL_DIAGNOSTICS = pd.DataFrame(well_rows)
for threshold in [150, 300, 600]:
    OOF[f"hybrid_{threshold}"] = np.where(
        OOF["well_id"].map(WELL_DIAGNOSTICS.set_index("well_id")["path_distance"]) <= threshold,
        OOF["neighbor_path_normalized"], OOF["ridge_prior"],
    )

CANDIDATES = [
    "last_known", "z_anchor", "ridge_prior",
    "neighbor_ps_normalized", "neighbor_path_normalized", "neighbor_path_md",
    "hybrid_150", "hybrid_300", "hybrid_600", "top5_oracle",
]
assert OOF["id"].is_unique and np.isfinite(OOF[CANDIDATES].to_numpy(float)).all()
print("OOF", OOF.shape)
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
    for candidate in ["ridge_prior", "neighbor_path_normalized", "neighbor_path_md", "top5_oracle"]:
        distance_rows.append({
            "distance_bin": str(distance_bin), "candidate": candidate,
            "wells": int(group["well_id"].nunique()), "rows": len(group),
            "pooled_rmse": rmse(group["target"], group[candidate]),
        })
DISTANCE_SCORES = pd.DataFrame(distance_rows)

display(SCORES)
display(DISTANCE_SCORES.pivot(index="distance_bin", columns="candidate", values="pooled_rmse"))
display(WELL_DIAGNOSTICS[["path_distance", "path_angle_difference_deg", "top5_oracle_rank"]].describe())
'''
)

md("## Takeaways and artifacts")

code(
    r'''SCORES.to_csv(WORK / "neighbor_scores_v1.csv", index=False)
DISTANCE_SCORES.to_csv(WORK / "neighbor_distance_scores_v1.csv", index=False)
WELL_DIAGNOSTICS.to_csv(WORK / "neighbor_well_diagnostics_v1.csv", index=False)

prediction_path = WORK / "neighbor_oof_predictions_v1.parquet"
OOF.to_parquet(prediction_path, index=False)

best_deployable = SCORES[SCORES["candidate"] != "top5_oracle"].iloc[0]
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
    "nearest_is_top5_oracle_share": float((WELL_DIAGNOSTICS["top5_oracle_rank"] == 1).mean()),
    "prediction_artifact": prediction_path.name,
    "caveats": [
        "Random well folds test access to nearby interpreted references; spatial holdout is not yet run.",
        "Top-5 oracle uses validation targets and is diagnostic only.",
        "No GR alignment, formation anchor, learned gate, or submission is included.",
    ],
}
(WORK / "neighbor_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
