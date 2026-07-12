from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CUTS = (0.20, 0.25, 0.33)
STRIDE = 8
BEAM_WIDTH = 16


def smooth(values, width=7):
    return (pd.Series(np.asarray(values, float)).interpolate(limit_direction="both")
            .rolling(width, center=True, min_periods=1).mean().to_numpy())


def beam_path(hw, tw, start_idx):
    tw = tw[["TVT", "GR"]].dropna().sort_values("TVT").drop_duplicates("TVT")
    tw_t, tw_g = tw.TVT.to_numpy(float), smooth(tw.GR)
    md, gr = hw.MD.to_numpy(float), smooth(hw.GR)
    known = hw.iloc[:start_idx + 1]
    recent = known.iloc[-min(300, len(known)):]
    dx, dy = np.diff(recent.MD), np.diff(recent.TVT)
    valid = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1e-9)
    slope = float(np.median(dy[valid] / dx[valid])) if valid.any() else 0.0
    idx = np.unique(np.r_[np.arange(start_idx, len(hw), STRIDE), len(hw) - 1]).astype(int)
    states, scores, paths = np.array([float(hw.TVT.iloc[start_idx])]), np.array([0.0]), [[]]
    for j in range(1, len(idx)):
        expected = states + slope * float(md[idx[j]] - md[idx[j - 1]])
        candidates = []
        for state_number, (base, exp) in enumerate(zip(scores, expected)):
            center = int(np.searchsorted(tw_t, exp))
            for typewell_idx in range(max(0, center - 5), min(len(tw_t), center + 6)):
                observation = 0.0 if not np.isfinite(gr[idx[j]]) else ((tw_g[typewell_idx] - gr[idx[j]]) / 25.0) ** 2
                motion = ((tw_t[typewell_idx] - exp) / 4.0) ** 2
                candidates.append((float(base + observation + motion), float(tw_t[typewell_idx]), state_number))
        candidates.sort(key=lambda row: row[0])
        chosen, seen = [], set()
        for item in candidates:
            key = round(item[1], 3)
            if key not in seen:
                chosen.append(item)
                seen.add(key)
            if len(chosen) >= BEAM_WIDTH:
                break
        paths = [paths[parent] + [state] for _, state, parent in chosen]
        scores = np.array([row[0] for row in chosen])
        states = np.array([row[1] for row in chosen])
    sparse = np.array([float(hw.TVT.iloc[start_idx])] + paths[int(np.argmin(scores))])
    return np.interp(np.arange(len(hw)), idx, sparse)


def prefix_backtest(hw, tw, outer_start):
    inner_start = max(50, int(round((outer_start + 1) * 0.75))) - 1
    path = beam_path(hw.iloc[:outer_start + 1].reset_index(drop=True), tw, inner_start)
    truth = hw.TVT.iloc[inner_start + 1:outer_start + 1].to_numpy(float)
    pred = path[inner_start + 1:outer_start + 1]
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def predictions(last, beam, prefix_rmse):
    n = len(beam)
    horizon = np.linspace(0.0, 1.0, n)

    def correction(limit, clip, start_weight, end_weight=None):
        if not np.isfinite(prefix_rmse) or prefix_rmse > limit:
            return np.full(n, last)
        confidence = max(0.0, 1.0 - prefix_rmse / limit)
        delta = np.clip(beam - last, -clip, clip)
        weights = confidence * (start_weight if end_weight is None else start_weight + (end_weight - start_weight) * horizon)
        return last + weights * delta

    return {
        "last_value": np.full(n, last),
        "v1_gate8_clip20": correction(8.0, 20.0, 0.15),
        "gate6_clip20": correction(6.0, 20.0, 0.15),
        "gate8_clip15": correction(8.0, 15.0, 0.15),
        "horizon_015_005": correction(8.0, 20.0, 0.15, 0.05),
    }


started = time.time()
rows, selector_cases, failures = [], [], []
files = sorted((DATA / "train").glob("*__horizontal_well.csv"))
for number, path in enumerate(files, 1):
    well = path.name.split("__")[0]
    try:
        hw = pd.read_csv(path).sort_values("MD").reset_index(drop=True)
        tw = pd.read_csv(DATA / "train" / f"{well}__typewell.csv")
        for cut in CUTS:
            start = int(round(len(hw) * cut)) - 1
            prefix_rmse = prefix_backtest(hw, tw, start)
            beam = beam_path(hw, tw, start)[start + 1:]
            truth = hw.TVT.iloc[start + 1:].to_numpy(float)
            last = float(hw.TVT.iloc[start])
            base_error = last - truth
            clipped_delta = np.clip(beam - last, -20.0, 20.0)
            selector_cases.append({
                "well": well,
                "cut": cut,
                "n_hidden": len(truth),
                "prefix_rmse": prefix_rmse,
                "prefix_fraction": float((start + 1) / len(hw)),
                "horizon_fraction": float(len(truth) / len(hw)),
                "gr_missing_rate": float(hw.GR.iloc[start + 1:].isna().mean()),
                "delta_mean_abs": float(np.mean(np.abs(clipped_delta))),
                "delta_p95_abs": float(np.quantile(np.abs(clipped_delta), 0.95)),
                "delta_max_abs": float(np.max(np.abs(clipped_delta))),
                "error_sq_sum": float(np.sum(base_error ** 2)),
                "error_delta_sum": float(np.sum(base_error * clipped_delta)),
                "delta_sq_sum": float(np.sum(clipped_delta ** 2)),
            })
            for model, pred in predictions(last, beam, prefix_rmse).items():
                error = pred - truth
                rows.append({
                    "well": well, "cut": cut, "model": model, "n_hidden": len(truth),
                    "prefix_rmse": prefix_rmse, "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "squared_error_sum": float(np.sum(error ** 2)),
                })
    except Exception as error:
        failures.append({"well": well, "error": repr(error)})
    if number % 100 == 0:
        print(f"{number}/{len(files)} wells, {time.time() - started:.0f}s", flush=True)

detail = pd.DataFrame(rows)
selector_df = pd.DataFrame(selector_cases)
summary_rows = []
for model, group in detail.groupby("model"):
    summary_rows.append({
        "model": model,
        "pooled_rmse": float(np.sqrt(group.squared_error_sum.sum() / group.n_hidden.sum())),
        "median_well_rmse": float(group.rmse.median()),
        "p90_well_rmse": float(group.rmse.quantile(0.90)),
        "worst_well_rmse": float(group.rmse.max()),
        "cases": len(group),
        "applied_cases": int((group.prefix_rmse <= (6.0 if model == "gate6_clip20" else 8.0)).sum()) if model != "last_value" else 0,
    })
summary = pd.DataFrame(summary_rows).sort_values("pooled_rmse")
detail.to_csv(RESULTS / "per_well_cut_metrics.csv", index=False)
selector_df.to_csv(RESULTS / "selector_cases.csv", index=False)
summary.to_csv(RESULTS / "summary.csv", index=False)
pd.DataFrame(failures).to_csv(RESULTS / "failures.csv", index=False)
run = {"wells": int(detail.well.nunique()), "cuts": list(CUTS), "elapsed_sec": time.time() - started,
       "best_model": str(summary.iloc[0].model), "best_pooled_rmse": float(summary.iloc[0].pooled_rmse)}
(RESULTS / "run.json").write_text(json.dumps(run, indent=2) + "\n")
print(summary.to_string(index=False))
print(json.dumps(run, indent=2))
