from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).with_name("rogii_candidate_oracle_audit_v1.ipynb")
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# ROGII candidate/oracle audit v1

Diagnostic only: this notebook never creates `submission.csv`.

It answers one question on all train wells using the organizer's real `TVT_input` prefix/suffix mask: **do we lack useful trajectory shapes, or do we already have them and fail to select them?**

Legal prediction inputs: `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`, paired typewell, and targets from other training wells. Train-only formation columns are intentionally ignored.

Artifacts:

- `candidate_scores.csv`
- `candidate_well_scores.csv`
- `candidate_diagnostics.csv`
- `candidate_oracle_summary.json`
- `oof_predictions.parquet` (or `.csv.gz` fallback)
"""
)

md(
    """## Experiment contract

- Five folds are assigned by whole well.
- A validation well can use its visible prefix but never its hidden suffix target.
- Neighbor candidates can use only wells outside the validation fold.
- Pooled per-point RMSE is the primary metric.
- Oracle shift/affine corrections are diagnostics only; they use suffix targets and are not deployable.
"""
)

code(
    r'''from pathlib import Path
import gc, json, os, time, warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

VERSION = "v1"
SEED = 42
N_SPLITS = 5
NEIGHBOR_INDEX_STRIDE = 20
NEIGHBOR_PROBE_STRIDE = 80
NEIGHBOR_PREFIX_ROWS = 400
GR_STRIDE = 5
GR_GRID = np.arange(-45.0, 45.01, 1.5)
GR_MAX_MOVE = 2
GR_MOVE_PENALTY = 0.35
SMOKE = os.environ.get("ROGII_SMOKE", "0") == "1"


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
        if root is None:
            continue
        if (root / "train").exists() and (root / "test").exists():
            return root
    raise FileNotFoundError("ROGII dataset root not found")


ROOT = find_root()
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
TRAIN_DIR = ROOT / "train"
HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
LEGAL_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]


def well_ids():
    return sorted(p.name.removesuffix(HORIZONTAL_SUFFIX) for p in TRAIN_DIR.glob(f"*{HORIZONTAL_SUFFIX}"))


def horizontal_path(well_id):
    return TRAIN_DIR / f"{well_id}{HORIZONTAL_SUFFIX}"


def typewell_path(well_id):
    return TRAIN_DIR / f"{well_id}{TYPEWELL_SUFFIX}"


def load_horizontal(well_id, include_target=True):
    columns = LEGAL_COLUMNS + (["TVT"] if include_target else [])
    return pd.read_csv(horizontal_path(well_id), usecols=columns)


def load_typewell(well_id):
    return pd.read_csv(typewell_path(well_id), usecols=["TVT", "GR"]).dropna().sort_values("TVT")


def split_mask(frame):
    known = frame["TVT_input"].notna().to_numpy()
    target = ~known
    return known, target


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


ids = well_ids()
usable = []
metadata = []
suffix_rows = 0
for well_id in ids:
    frame = load_horizontal(well_id)
    known, target = split_mask(frame)
    if known.sum() < 20 or target.sum() < 20 or not np.isfinite(frame.loc[target, "TVT"]).all():
        continue
    usable.append(well_id)
    suffix_rows += int(target.sum())
    dx = float(frame["X"].iloc[-1] - frame["X"].iloc[0])
    dy = float(frame["Y"].iloc[-1] - frame["Y"].iloc[0])
    metadata.append({
        "well_id": well_id,
        "azimuth_deg": float(np.degrees(np.arctan2(dy, dx))),
        "direction": "SE_like" if dx >= 0 else "NW_like",
        "suffix_rows": int(target.sum()),
    })

if SMOKE:
    usable = usable[:20]
    metadata = [row for row in metadata if row["well_id"] in set(usable)]
    N_SPLITS = 2

META = pd.DataFrame(metadata).set_index("well_id")
rng = np.random.default_rng(SEED)
shuffled = np.array(usable, dtype=object)
rng.shuffle(shuffled)
FOLD_BY_WELL = {well_id: int(i % N_SPLITS) for i, well_id in enumerate(shuffled)}

print({
    "version": VERSION,
    "root": str(ROOT),
    "wells": len(usable),
    "suffix_rows_full": suffix_rows,
    "folds": N_SPLITS,
    "smoke": SMOKE,
})
'''
)

md("## Same-well geometry and GR candidates")

code(
    r'''def tail_linear_u(frame, known, target):
    known_idx = np.flatnonzero(known)
    tail_idx = known_idx[-min(400, len(known_idx)):]
    md = frame["MD"].to_numpy(float)
    z = frame["Z"].to_numpy(float)
    tvt_input = frame["TVT_input"].to_numpy(float)
    u = tvt_input[tail_idx] + z[tail_idx]
    x = md[tail_idx] - md[tail_idx[-1]]
    slope = float(np.polyfit(x, u, 1)[0]) if len(tail_idx) >= 3 and np.ptp(x) > 0 else 0.0
    # ponytail: bounded linear structural prior; replace with a learned curvature prior if this ceiling is too high.
    slope = float(np.clip(slope, -0.03, 0.03))
    u_last = float(np.median(u[-min(80, len(u)):]))
    target_idx = np.flatnonzero(target)
    u_hold = np.full(len(target_idx), u_last) - z[target_idx]
    u_linear = u_last + slope * (md[target_idx] - md[tail_idx[-1]]) - z[target_idx]
    return u_hold, u_linear


def calibrate_typewell_gr(frame, typewell, known):
    tvt_grid = typewell["TVT"].to_numpy(float)
    gr_grid = typewell["GR"].to_numpy(float)
    observed = frame["GR"].to_numpy(float)
    tvt_input = frame["TVT_input"].to_numpy(float)
    mask = known & np.isfinite(observed) & np.isfinite(tvt_input)
    if mask.sum() < 20:
        return tvt_grid, gr_grid, 25.0
    predicted = np.interp(tvt_input[mask], tvt_grid, gr_grid)
    design = np.c_[predicted, np.ones(mask.sum())]
    coef = np.linalg.lstsq(design, observed[mask], rcond=None)[0]
    scale = float(np.clip(coef[0], 0.3, 3.0))
    bias = float(coef[1])
    calibrated = scale * gr_grid + bias
    residual = observed[mask] - np.interp(tvt_input[mask], tvt_grid, calibrated)
    sigma = float(np.clip(np.nanmedian(np.abs(residual - np.nanmedian(residual))) * 1.4826, 8.0, 50.0))
    return tvt_grid, calibrated, sigma


def rolling_mean(values, window):
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy(float)


def viterbi_gr(frame, typewell, target, prior):
    target_idx = np.flatnonzero(target)
    sampled_pos = np.arange(0, len(target_idx), GR_STRIDE)
    if sampled_pos[-1] != len(target_idx) - 1:
        sampled_pos = np.r_[sampled_pos, len(target_idx) - 1]
    row_idx = target_idx[sampled_pos]

    observed = frame["GR"].astype(float).interpolate(limit_direction="both").to_numpy(float)[row_idx]
    if not np.isfinite(observed).any():
        return prior.copy()
    observed = pd.Series(observed).fillna(float(np.nanmedian(observed))).to_numpy(float)
    known, _ = split_mask(frame)
    tvt_grid, type_gr, sigma = calibrate_typewell_gr(frame, typewell, known)
    type_gr_smooth = rolling_mean(type_gr, 21)
    observed_smooth = rolling_mean(observed, 9)

    states = prior[sampled_pos, None] + GR_GRID[None, :]
    expected_raw = np.interp(states, tvt_grid, type_gr)
    expected_smooth = np.interp(states, tvt_grid, type_gr_smooth)
    raw_cost = ((observed[:, None] - expected_raw) / sigma) ** 2
    smooth_cost = ((observed_smooth[:, None] - expected_smooth) / sigma) ** 2
    emission = np.minimum(0.5 * raw_cost + 0.5 * smooth_cost, 25.0)

    n_time, n_state = emission.shape
    center = int(np.argmin(np.abs(GR_GRID)))
    dp = np.full(n_state, 25.0)
    dp[center] = 0.0
    back = np.zeros((n_time, n_state), np.int8)
    shifts = np.arange(-GR_MAX_MOVE, GR_MAX_MOVE + 1)
    for t in range(n_time):
        alternatives = np.full((len(shifts), n_state), np.inf)
        for si, shift in enumerate(shifts):
            if shift < 0:
                alternatives[si, :shift] = dp[-shift:] + GR_MOVE_PENALTY * shift * shift
            elif shift > 0:
                alternatives[si, shift:] = dp[:-shift] + GR_MOVE_PENALTY * shift * shift
            else:
                alternatives[si] = dp
        choice = np.argmin(alternatives, axis=0)
        dp = alternatives[choice, np.arange(n_state)] + emission[t]
        dp -= float(dp.min())
        back[t] = shifts[choice]

    path = np.empty(n_time, dtype=int)
    path[-1] = int(np.argmin(dp))
    for t in range(n_time - 1, 0, -1):
        path[t - 1] = int(np.clip(path[t] - back[t, path[t]], 0, n_state - 1))
    sampled_prediction = prior[sampled_pos] + GR_GRID[path]
    return np.interp(np.arange(len(target_idx)), sampled_pos, sampled_prediction)


def same_well_candidates(frame, typewell):
    known, target = split_mask(frame)
    target_idx = np.flatnonzero(target)
    last_known = float(frame.loc[known, "TVT_input"].iloc[-1])
    u_hold, u_linear = tail_linear_u(frame, known, target)
    return target_idx, {
        "last_known": np.full(len(target_idx), last_known),
        "u_hold": u_hold,
        "u_linear": u_linear,
        "gr_viterbi": viterbi_gr(frame, typewell, target, u_linear),
    }
'''
)

md("## Fold-safe XY neighbor transfer")

code(
    r'''def build_spatial_index(reference_wells):
    points = []
    owners = []
    for owner_index, well_id in enumerate(reference_wells):
        frame = load_horizontal(well_id)
        xy = frame[["X", "Y"]].to_numpy(float)[::NEIGHBOR_INDEX_STRIDE]
        points.append(xy)
        owners.append(np.full(len(xy), owner_index, np.int32))
    points = np.concatenate(points)
    owners = np.concatenate(owners)
    return cKDTree(points), owners, list(reference_wells)


def choose_neighbor(frame, global_tree, owners, reference_wells):
    probe_xy = frame[["X", "Y"]].to_numpy(float)[::NEIGHBOR_PROBE_STRIDE]
    _, global_index = global_tree.query(probe_xy, k=8)
    candidate_owner_indices = np.unique(owners[np.asarray(global_index).reshape(-1)])
    best = None
    for owner_index in candidate_owner_indices:
        well_id = reference_wells[int(owner_index)]
        reference = load_horizontal(well_id)
        reference_xy = reference[["X", "Y"]].to_numpy(float)[::NEIGHBOR_INDEX_STRIDE]
        distance = cKDTree(reference_xy).query(probe_xy, k=1)[0]
        score = float(np.median(distance))
        if best is None or score < best[0]:
            best = (score, well_id)
    if best is None:
        raise RuntimeError("no neighbor candidate found")
    return best


def neighbor_prediction(frame, target, neighbor_id):
    reference = load_horizontal(neighbor_id)
    reference_xy = reference[["X", "Y"]].to_numpy(float)
    tree = cKDTree(reference_xy)
    target_xy = frame[["X", "Y"]].to_numpy(float)
    distance, nearest = tree.query(target_xy, k=1)
    mapped = reference["TVT"].to_numpy(float)[nearest]

    known_idx = np.flatnonzero(~target)[-NEIGHBOR_PREFIX_ROWS:]
    valid = np.isfinite(frame["TVT_input"].to_numpy(float)[known_idx]) & np.isfinite(mapped[known_idx])
    offset = float(np.median(frame["TVT_input"].to_numpy(float)[known_idx][valid] - mapped[known_idx][valid])) if valid.any() else 0.0
    target_idx = np.flatnonzero(target)
    return mapped[target_idx] + offset, float(np.median(distance[::NEIGHBOR_PROBE_STRIDE]))
'''
)

md("## Five-fold OOF candidate bank")

code(
    r'''candidate_frames = []
start_time = time.time()

for fold in range(N_SPLITS):
    valid_wells = [well_id for well_id in usable if FOLD_BY_WELL[well_id] == fold]
    reference_wells = [well_id for well_id in usable if FOLD_BY_WELL[well_id] != fold]
    global_tree, owners, indexed_wells = build_spatial_index(reference_wells)
    print(f"fold={fold} valid={len(valid_wells)} refs={len(reference_wells)} index_points={global_tree.n}")

    for position, well_id in enumerate(valid_wells, 1):
        frame = load_horizontal(well_id)
        target_values = frame["TVT"].to_numpy(float)
        legal_frame = frame[LEGAL_COLUMNS].copy()
        typewell = load_typewell(well_id)
        _, target = split_mask(legal_frame)
        target_idx, candidates = same_well_candidates(legal_frame, typewell)
        _, neighbor_id = choose_neighbor(legal_frame, global_tree, owners, indexed_wells)
        candidates["neighbor_xy"], mapped_distance = neighbor_prediction(legal_frame, target, neighbor_id)

        result = pd.DataFrame({
            "id": [f"{well_id}_{row}" for row in target_idx],
            "well_id": well_id,
            "row_index": target_idx,
            "fold": fold,
            "target": target_values[target_idx],
            "md": legal_frame["MD"].to_numpy(float)[target_idx],
            "neighbor_id": neighbor_id,
            "neighbor_distance": mapped_distance,
            "azimuth_deg": float(META.loc[well_id, "azimuth_deg"]),
            "direction": str(META.loc[well_id, "direction"]),
            **candidates,
        })
        candidate_frames.append(result)
        if position % 25 == 0 or position == len(valid_wells):
            print(f"  {position}/{len(valid_wells)} elapsed={time.time() - start_time:.0f}s")

    del global_tree, owners
    gc.collect()

OOF = pd.concat(candidate_frames, ignore_index=True)
CANDIDATES = ["last_known", "u_hold", "u_linear", "gr_viterbi", "neighbor_xy"]

assert OOF["id"].is_unique
assert OOF[CANDIDATES].notna().all().all()
assert np.isfinite(OOF[CANDIDATES].to_numpy(float)).all()
assert all(
    FOLD_BY_WELL[neighbor_id] != fold
    for neighbor_id, fold in OOF[["neighbor_id", "fold"]].drop_duplicates().itertuples(index=False, name=None)
)
print("OOF", OOF.shape, "elapsed", round(time.time() - start_time, 1), "seconds")
'''
)

md("## Scores, shape ceilings, and oracle diagnostics")

code(
    r'''def per_well_oracle_corrections(group, candidate):
    y = group["target"].to_numpy(float)
    p = group[candidate].to_numpy(float)
    x = group["md"].to_numpy(float)
    x = (x - x.mean()) / max(x.std(), 1e-9)
    error = y - p
    shift = float(error.mean())
    affine = np.linalg.lstsq(np.c_[np.ones(len(x)), x], error, rcond=None)[0]
    return (
        float(np.sum((y - (p + shift)) ** 2)),
        float(np.sum((y - (p + affine[0] + affine[1] * x)) ** 2)),
    )


score_rows = []
well_rows = []
for candidate in CANDIDATES:
    shift_sse = 0.0
    affine_sse = 0.0
    for well_id, group in OOF.groupby("well_id", sort=False):
        score = rmse(group["target"], group[candidate])
        well_rows.append({
            "well_id": well_id,
            "candidate": candidate,
            "rmse": score,
            "rows": len(group),
            "fold": int(group["fold"].iloc[0]),
            "neighbor_distance": float(group["neighbor_distance"].iloc[0]),
            "direction": str(group["direction"].iloc[0]),
        })
        shifted, affine = per_well_oracle_corrections(group, candidate)
        shift_sse += shifted
        affine_sse += affine
    score_rows.append({
        "candidate": candidate,
        "pooled_rmse": rmse(OOF["target"], OOF[candidate]),
        "oracle_shift_rmse": float(np.sqrt(shift_sse / len(OOF))),
        "oracle_affine_rmse": float(np.sqrt(affine_sse / len(OOF))),
    })

SCORES = pd.DataFrame(score_rows).sort_values("pooled_rmse").reset_index(drop=True)
WELL_SCORES = pd.DataFrame(well_rows)

shape_sse = {0: 0.0, 1: 0.0, 2: 0.0}
well_oracle_sse = 0.0
row_oracle_sse = 0.0
for well_id, group in OOF.groupby("well_id", sort=False):
    y = group["target"].to_numpy(float)
    x = group["md"].to_numpy(float)
    x = (x - x.mean()) / max(x.std(), 1e-9)
    for degree in shape_sse:
        fitted = np.polyval(np.polyfit(x, y, degree), x)
        shape_sse[degree] += float(np.sum((y - fitted) ** 2))
    errors = np.stack([(y - group[candidate].to_numpy(float)) ** 2 for candidate in CANDIDATES], axis=1)
    well_oracle_sse += float(errors.sum(axis=0).min())
    row_oracle_sse += float(errors.min(axis=1).sum())

SHAPE_CEILINGS = {f"degree_{degree}": float(np.sqrt(sse / len(OOF))) for degree, sse in shape_sse.items()}
ORACLES = {
    "well_candidate_oracle_rmse": float(np.sqrt(well_oracle_sse / len(OOF))),
    "row_candidate_oracle_rmse": float(np.sqrt(row_oracle_sse / len(OOF))),
}

if not SMOKE:
    assert 8.5 < SHAPE_CEILINGS["degree_0"] < 9.5
    assert 6.2 < SHAPE_CEILINGS["degree_1"] < 7.2
    assert 4.8 < SHAPE_CEILINGS["degree_2"] < 5.8

display(SCORES)
print("shape ceilings", SHAPE_CEILINGS)
print("candidate oracles", ORACLES)
'''
)

md("## Distance and direction diagnostics")

code(
    r'''OOF["distance_bin"] = pd.cut(
    OOF["neighbor_distance"],
    bins=[-np.inf, 150.0, 600.0, np.inf],
    labels=["under_150", "150_to_600", "over_600"],
)

diagnostic_rows = []
for dimension in ["distance_bin", "direction", "fold"]:
    for value, group in OOF.groupby(dimension, observed=True):
        for candidate in CANDIDATES:
            diagnostic_rows.append({
                "dimension": dimension,
                "value": str(value),
                "candidate": candidate,
                "rows": len(group),
                "wells": int(group["well_id"].nunique()),
                "pooled_rmse": rmse(group["target"], group[candidate]),
            })

DIAGNOSTICS = pd.DataFrame(diagnostic_rows)
display(DIAGNOSTICS[DIAGNOSTICS["dimension"] == "distance_bin"].pivot(index="candidate", columns="value", values="pooled_rmse"))
display(DIAGNOSTICS[DIAGNOSTICS["dimension"] == "direction"].pivot(index="candidate", columns="value", values="pooled_rmse"))
'''
)

md("## Persist auditable artifacts")

code(
    r'''SCORES.to_csv(WORK / "candidate_scores.csv", index=False)
WELL_SCORES.to_csv(WORK / "candidate_well_scores.csv", index=False)
DIAGNOSTICS.to_csv(WORK / "candidate_diagnostics.csv", index=False)

prediction_columns = [
    "id", "well_id", "row_index", "fold", "target", "md", "neighbor_id",
    "neighbor_distance", "azimuth_deg", "direction", "distance_bin", *CANDIDATES,
]
prediction_path = WORK / "oof_predictions.parquet"
try:
    OOF[prediction_columns].to_parquet(prediction_path, index=False)
except Exception as error:
    prediction_path = WORK / "oof_predictions.csv.gz"
    OOF[prediction_columns].to_csv(prediction_path, index=False, compression="gzip")
    print("parquet fallback:", error)

summary = {
    "version": VERSION,
    "legal_columns": LEGAL_COLUMNS,
    "wells": int(OOF["well_id"].nunique()),
    "rows": int(len(OOF)),
    "scores": SCORES.to_dict("records"),
    "shape_ceilings": SHAPE_CEILINGS,
    "candidate_oracles": ORACLES,
    "prediction_artifact": prediction_path.name,
    "decision_rule": {
        "generator_problem": "well candidate oracle remains above 6.0",
        "selector_problem": "well candidate oracle reaches 5.5 or below",
    },
}
(WORK / "candidate_oracle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("artifacts written to", WORK)
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
