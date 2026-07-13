#!/usr/bin/env python3
"""Build the public-facing Biohub data atlas research notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "biohub_data_atlas_validation_research.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


cells = [
    markdown(r"""
<div style="padding:34px 38px;border-radius:22px;background:linear-gradient(135deg,#071A2E 0%,#123B58 58%,#176B87 100%);color:white;margin-bottom:22px">
  <div style="font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:#9ED9E8;font-weight:700">Biohub · 3D Cell Tracking Research</div>
  <h1 style="font-size:38px;line-height:1.12;margin:12px 0 10px;color:white">Data Atlas & Validation Research</h1>
  <p style="font-size:17px;line-height:1.55;max-width:900px;color:#DCEEF4;margin:0">A reproducible atlas of imaging conditions, lineage structure, cell motion, annotation coverage, and embryo-level validation design.</p>
</div>

> **Purpose.** Discover where the tracking problem changes across embryos and build evidence for a validation strategy that can survive the private leaderboard.
"""),
    markdown(r"""
## tl;dr

This notebook produces an evidence package rather than a submission:

1. a catalog of every available train/test volume;
2. a structural profile of labeled lineage graphs;
3. motion, track-length, density, and division diagnostics;
4. train/test shift indicators that do not use test labels;
5. balanced embryo-level folds for honest downstream experiments.

The summary cards and findings below are generated from executed data. No result is hard-coded.
"""),
    markdown(r"""
## Context & Methods

### Key assumptions

- A dataset name prefix before the first underscore identifies an embryo group. This is visible and easy to replace in `embryo_id()`.
- GEFF annotations may be sparse; therefore `node_recall` and annotated node counts are descriptors, not complete biological cell counts.
- Test images are used only for label-free distribution comparisons.
- `FAST_MODE=True` profiles a deterministic subset of labeled graphs while cataloging all image volumes.

### Research questions

- Do train and test differ in volume size, intensity range, or temporal length?
- Which embryos contain the densest motion, shortest tracks, and most divisions?
- Are the public model's global distance and filtering thresholds biologically plausible?
- Can grouped folds balance embryos, annotated nodes, edges, and division events?
"""),
    code(r"""
from pathlib import Path
from IPython.display import display, HTML
import json, math, os, shutil, subprocess, sys, textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR = Path(f"/kaggle/input/competitions/{COMPETITION}")
TRAIN_DIR = COMP_DIR / "train"
TEST_DIR = COMP_DIR / "test"
WORK_DIR = Path("/kaggle/working/biohub_research")

FAST_MODE = True
MAX_LABELED_DATASETS = 24 if FAST_MODE else None
N_FOLDS = 5
RANDOM_SEED = 42

assert TRAIN_DIR.exists(), f"Attach competition data: {TRAIN_DIR}"
WORK_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#123B58", "blue": "#176B87", "cyan": "#64C4D2",
    "gold": "#E3A72F", "ink": "#17222D", "muted": "#667784",
    "grid": "#DCE5EA", "paper": "#F6F9FA",
}
plt.rcParams.update({
    "figure.figsize": (11.5, 5.2), "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["ink"], "axes.titlecolor": COLORS["ink"],
    "axes.titlesize": 15, "axes.titleweight": "bold", "font.size": 11,
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "grid.color": COLORS["grid"], "grid.alpha": .72,
})
np.random.seed(RANDOM_SEED)
print(f"Mode: {'FAST' if FAST_MODE else 'FULL'} · train={TRAIN_DIR} · test_exists={TEST_DIR.exists()}")
"""),
    markdown("### 1. Prepare the analysis runtime"),
    code(r"""
# Use the support artifact's tested wheel snapshot in a private directory.
manifest_candidates = sorted(Path("/kaggle/input").glob("**/ARTIFACT_MANIFEST.json"))
support_manifest = None
for candidate in manifest_candidates:
    try:
        manifest = json.loads(candidate.read_text())
    except Exception:
        continue
    if manifest.get("model", {}).get("method") == "unet_transformer" and (candidate.parent / "wheels").exists():
        support_manifest = candidate
        break
assert support_manifest, "Attach pilkwang/biohub-tracking-support-pack-50ep-v1"

SUPPORT_DIR = support_manifest.parent
RUNTIME_SITE = WORK_DIR / "runtime_site"
shutil.rmtree(RUNTIME_SITE, ignore_errors=True)
RUNTIME_SITE.mkdir(parents=True)
wheel_files = sorted((SUPPORT_DIR / "wheels").glob("*.whl"))
installed = subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
    "--upgrade", "--force-reinstall", "--target", str(RUNTIME_SITE), *map(str, wheel_files),
], text=True, capture_output=True)
assert installed.returncode == 0, installed.stderr[-12000:]

runtime_env = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join([str(RUNTIME_SITE), str(SUPPORT_DIR / "repo" / "src")]),
    "PYTHONNOUSERSITE": "1",
}
probe = subprocess.run([
    sys.executable, "-c", "import tracksdata, geff, zarr, numcodecs; print('runtime ready')"
], env=runtime_env, text=True, capture_output=True)
assert probe.returncode == 0, probe.stderr[-12000:]
print(probe.stdout.strip(), "· artifact:", json.loads(support_manifest.read_text()).get("artifact_name"))
"""),
    markdown(r"""
## Data

### 2. Build the image and lineage catalog

The engine reads lightweight Zarr metadata for every volume. In fast mode it then samples labeled GEFF graphs deterministically across the full sorted dataset list.
"""),
    code(r'''
ANALYZER = r"""
from pathlib import Path
from collections import Counter, defaultdict, deque
import argparse, csv, json, math, statistics

import tracksdata as td
from geff import GeffMetadata

DEFAULT_SCALE = (1.625, 0.40625, 0.40625)

def read_json(path):
    try: return json.loads(path.read_text())
    except Exception: return {}

def embryo_id(name):
    return name.split('_', 1)[0]

def scale_and_quantiles(root):
    scale = DEFAULT_SCALE
    try:
        transforms = root['attributes']['multiscales'][0]['datasets'][0]['coordinateTransformations']
        for transform in transforms:
            if transform.get('type') == 'scale': scale = tuple(map(float, transform['scale'][-3:]))
    except Exception: pass
    q = root.get('attributes', {}).get('image_statistics', {}).get('quantiles', {}) or {}
    return scale, q

def load_graph(path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded

def write_rows(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=Path, required=True)
parser.add_argument('--test', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--max-labeled', type=int, default=0)
args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

catalog = []
for split, folder in [('train', args.train), ('test', args.test)]:
    if not folder.exists(): continue
    for zarr_path in sorted(folder.glob('*.zarr')):
        arr = read_json(zarr_path / '0' / 'zarr.json')
        root = read_json(zarr_path / 'zarr.json')
        shape = arr.get('shape', [None] * 4)
        scale, q = scale_and_quantiles(root)
        geff_path = folder / f'{zarr_path.stem}.geff'
        estimated = None
        if geff_path.exists():
            try: estimated = (GeffMetadata.read(geff_path).extra or {}).get('estimated_number_of_nodes')
            except Exception: pass
        catalog.append({
            'dataset': zarr_path.stem, 'embryo': embryo_id(zarr_path.stem), 'split': split,
            'T': shape[0], 'Z': shape[1], 'Y': shape[2], 'X': shape[3],
            'spatial_mvox': (shape[1] * shape[2] * shape[3] / 1e6) if all(v is not None for v in shape[1:]) else None,
            'scale_z': scale[0], 'scale_y': scale[1], 'scale_x': scale[2],
            'q001': q.get('0.001'), 'q500': q.get('0.5'), 'q999': q.get('0.999'),
            'estimated_nodes': estimated, 'has_labels': int(geff_path.exists()),
        })

train_names = [r['dataset'] for r in catalog if r['split'] == 'train' and r['has_labels']]
if args.max_labeled and len(train_names) > args.max_labeled:
    if args.max_labeled == 1: indices = [0]
    else: indices = [round(i * (len(train_names) - 1) / (args.max_labeled - 1)) for i in range(args.max_labeled)]
    train_names = [train_names[i] for i in sorted(set(indices))]

profiles, time_rows, motion_rows, track_rows, division_rows = [], [], [], [], []
for name in train_names:
    graph = load_graph(args.train / f'{name}.geff')
    nodes = {int(r['node_id']): r for r in graph.node_attrs().iter_rows(named=True)}
    edges = [(int(r['source_id']), int(r['target_id'])) for r in graph.edge_attrs().iter_rows(named=True)]
    row = next(r for r in catalog if r['dataset'] == name and r['split'] == 'train')
    scale = (float(row['scale_z']), float(row['scale_y']), float(row['scale_x']))
    by_t = Counter(int(n['t']) for n in nodes.values())
    for t, count in sorted(by_t.items()): time_rows.append({'dataset': name, 'embryo': embryo_id(name), 't': t, 'annotated_nodes': count})

    adjacency = defaultdict(set); out_targets = defaultdict(list)
    for source, target in edges:
        adjacency[source].add(target); adjacency[target].add(source); out_targets[source].append(target)
    seen = set(); lengths = []
    for node_id in nodes:
        if node_id in seen: continue
        queue = [node_id]; seen.add(node_id); size = 0
        while queue:
            current = queue.pop(); size += 1
            for nxt in adjacency[current]:
                if nxt not in seen: seen.add(nxt); queue.append(nxt)
        lengths.append(size); track_rows.append({'dataset': name, 'embryo': embryo_id(name), 'track_nodes': size})

    distances = []
    stride = max(1, len(edges) // 5000)
    for source, target in edges[::stride]:
        if source not in nodes or target not in nodes: continue
        a, b = nodes[source], nodes[target]
        d = math.sqrt(sum(((float(a[k]) - float(b[k])) * s) ** 2 for k, s in zip(('z','y','x'), scale)))
        distances.append(d); motion_rows.append({'dataset': name, 'embryo': embryo_id(name), 'distance_um': d})

    divisions = 0
    for source, targets in out_targets.items():
        if len(targets) != 2 or source not in nodes or any(t not in nodes for t in targets): continue
        divisions += 1; parent = nodes[source]; children = [nodes[t] for t in targets]
        parent_ds = [math.sqrt(sum(((float(parent[k])-float(c[k]))*s)**2 for k,s in zip(('z','y','x'),scale))) for c in children]
        sister = math.sqrt(sum(((float(children[0][k])-float(children[1][k]))*s)**2 for k,s in zip(('z','y','x'),scale)))
        division_rows.append({'dataset': name, 'embryo': embryo_id(name), 'parent_max_um': max(parent_ds), 'sister_um': sister})

    profiles.append({
        'dataset': name, 'embryo': embryo_id(name), 'annotated_nodes': len(nodes), 'edges': len(edges),
        'divisions': divisions, 'components': len(lengths),
        'median_track_nodes': statistics.median(lengths) if lengths else None,
        'median_motion_um': statistics.median(distances) if distances else None,
        'p95_motion_um': sorted(distances)[min(len(distances)-1, int(.95*len(distances)))] if distances else None,
        'mean_nodes_per_frame': statistics.mean(by_t.values()) if by_t else None,
    })

write_rows(args.out/'catalog.csv', list(catalog[0]), catalog)
tables = [
 ('profiles.csv', profiles), ('time_series.csv', time_rows), ('motion.csv', motion_rows),
 ('track_lengths.csv', track_rows), ('divisions.csv', division_rows),
]
for filename, rows in tables:
    fields = list(rows[0]) if rows else ['dataset']
    write_rows(args.out/filename, fields, rows)
print(json.dumps({'catalog': len(catalog), 'profiled': len(profiles), 'motion_edges': len(motion_rows), 'divisions': len(division_rows)}))
"""

engine_path = WORK_DIR / "analyze_dataset.py"
engine_path.write_text(ANALYZER)
max_labeled = str(MAX_LABELED_DATASETS or 0)
analysis = subprocess.run([
    sys.executable, str(engine_path), "--train", str(TRAIN_DIR), "--test", str(TEST_DIR),
    "--out", str(WORK_DIR), "--max-labeled", max_labeled,
], env=runtime_env, text=True, capture_output=True)
assert analysis.returncode == 0, analysis.stderr[-12000:]
print(analysis.stdout.strip())
'''),
    code(r"""
def read_table(name, required=()):
    path = WORK_DIR / name
    frame = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    for column in required:
        if column not in frame: frame[column] = pd.Series(dtype="float64")
    return frame

catalog = read_table("catalog.csv")
profiles = read_table("profiles.csv", ["annotated_nodes", "edges", "divisions"])
time_series = read_table("time_series.csv", ["t", "annotated_nodes"])
motion = read_table("motion.csv", ["distance_um"])
track_lengths = read_table("track_lengths.csv", ["track_nodes"])
divisions = read_table("divisions.csv", ["parent_max_um", "sister_um"])

for col in ["T","Z","Y","X","spatial_mvox","q001","q500","q999","estimated_nodes"]:
    if col in catalog: catalog[col] = pd.to_numeric(catalog[col], errors="coerce")

assert len(catalog) > 0 and set(catalog["split"]) <= {"train", "test"}
assert catalog["dataset"].notna().all()
print(f"Catalog: {len(catalog):,} volumes · profiled graphs: {len(profiles):,}")
display(catalog.head(8))
"""),
    markdown("## Results\n\n### 3. Dataset coverage at a glance"),
    code(r"""
def cards(items):
    blocks = "".join(
        f'<div style="flex:1;min-width:150px;padding:18px 20px;border:1px solid #DCE5EA;border-radius:16px;background:#F8FBFC">'
        f'<div style="font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#667784;font-weight:700">{label}</div>'
        f'<div style="font-size:28px;color:#123B58;font-weight:800;margin-top:6px">{value}</div></div>'
        for label, value in items
    )
    display(HTML(f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 20px">{blocks}</div>'))

cards([
    ("Train volumes", f"{(catalog.split == 'train').sum():,}"),
    ("Test volumes", f"{(catalog.split == 'test').sum():,}"),
    ("Embryos", f"{catalog.embryo.nunique():,}"),
    ("Profiled graphs", f"{len(profiles):,}"),
    ("Observed divisions", f"{int(profiles.divisions.sum()) if len(profiles) else 0:,}"),
])

coverage = catalog.groupby("split", as_index=False).agg(
    volumes=("dataset", "size"), embryos=("embryo", "nunique"),
    median_frames=("T", "median"), median_spatial_mvox=("spatial_mvox", "median"),
)
display(coverage.style.format({"median_frames":"{:.0f}", "median_spatial_mvox":"{:.2f}"}).hide(axis="index"))
"""),
    markdown("### 4. Imaging regimes and label-free train/test shift"),
    code(r"""
plot_data = catalog.dropna(subset=["T", "spatial_mvox"])
fig, ax = plt.subplots(figsize=(10.8, 5.4))
for split, color, marker in [("train", COLORS["blue"], "o"), ("test", COLORS["gold"], "D")]:
    part = plot_data[plot_data.split == split]
    if len(part): ax.scatter(part["T"], part.spatial_mvox, s=52, alpha=.68, c=color, marker=marker, label=f"{split} (n={len(part)})", edgecolors="white", linewidths=.5)
ax.set(title="Volume geometry across available datasets", xlabel="Frames per video (T)", ylabel="Spatial volume per frame (million voxels)")
ax.grid(True); ax.legend(frameon=False); plt.show()

shift_rows = []
for feature in ["T", "spatial_mvox", "q001", "q500", "q999"]:
    train = pd.to_numeric(catalog.loc[catalog.split == "train", feature], errors="coerce").dropna()
    test = pd.to_numeric(catalog.loc[catalog.split == "test", feature], errors="coerce").dropna()
    if len(train) and len(test):
        pooled = pd.concat([train, test]).std()
        shift_rows.append({"feature": feature, "train_median": train.median(), "test_median": test.median(), "median_gap_in_pooled_sd": (test.median()-train.median())/pooled if pooled else 0})
shift = pd.DataFrame(shift_rows)
display(shift.style.format({"train_median":"{:.3g}","test_median":"{:.3g}","median_gap_in_pooled_sd":"{:+.2f}"}).hide(axis="index") if len(shift) else HTML("<i>No comparable test metadata found.</i>"))
"""),
    markdown("### 5. Lineage graph structure"),
    code(r"""
if len(profiles):
    ordered = profiles.sort_values("annotated_nodes", ascending=True).tail(18)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.barh(ordered.dataset, ordered.annotated_nodes, color=COLORS["blue"], alpha=.9)
    ax.set(title="Largest profiled annotation graphs", xlabel="Annotated nodes", ylabel="Dataset")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False); plt.show()

    display(profiles.describe(percentiles=[.1,.5,.9,.95]).T[["count","mean","50%","90%","95%","max"]].style.format("{:,.2f}"))
else:
    display(HTML("<div style='padding:16px;border-radius:12px;background:#FFF5DC'>No labeled graphs were profiled.</div>"))
"""),
    markdown("### 6. Temporal annotation density"),
    code(r"""
if len(time_series):
    focus = profiles.sort_values("annotated_nodes", ascending=False).iloc[0].dataset
    series = time_series[time_series.dataset == focus].sort_values("t")
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.plot(series.t, series.annotated_nodes, color=COLORS["blue"], lw=2.4)
    ax.fill_between(series.t, series.annotated_nodes, color=COLORS["cyan"], alpha=.2)
    ax.set(title=f"Annotated cells through time · {focus}", xlabel="Frame", ylabel="Annotated nodes")
    ax.grid(True); plt.show()
else: print("No time-series rows available.")
"""),
    markdown("### 7. Motion scale and candidate linking radii"),
    code(r"""
distances = pd.to_numeric(motion.distance_um, errors="coerce").dropna()
if len(distances):
    upper = distances.quantile(.995)
    shown = distances[distances <= upper]
    p50, p90, p95, p99 = [distances.quantile(q) for q in [.5,.9,.95,.99]]
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.hist(shown, bins=55, color=COLORS["blue"], alpha=.88, edgecolor="white")
    for value, label, color in [(p90,"p90",COLORS["gold"]),(p95,"p95",COLORS["cyan"]),(p99,"p99",COLORS["ink"])]:
        ax.axvline(value, color=color, ls="--", lw=2, label=f"{label} {value:.2f} µm")
    ax.set(title="Observed annotated edge displacement", xlabel="Parent → child displacement (µm)", ylabel="Edges")
    ax.legend(frameon=False); ax.grid(axis="y"); plt.show()
    cards([("Median motion", f"{p50:.2f} µm"), ("p95 motion", f"{p95:.2f} µm"), ("p99 motion", f"{p99:.2f} µm"), ("Edges sampled", f"{len(distances):,}")])
else: print("No motion observations available.")
"""),
    markdown("### 8. Track components and division geometry"),
    code(r"""
lengths = pd.to_numeric(track_lengths.track_nodes, errors="coerce").dropna()
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
if len(lengths):
    cap = max(2, lengths.quantile(.99))
    axes[0].hist(lengths.clip(upper=cap), bins=40, color=COLORS["blue"], edgecolor="white")
    axes[0].set(title="Connected lineage component sizes", xlabel=f"Nodes per component (clipped at p99={cap:.0f})", ylabel="Components")
else: axes[0].text(.5,.5,"No track components",ha="center",va="center")

if len(divisions):
    axes[1].scatter(divisions.parent_max_um, divisions.sister_um, s=30, alpha=.55, c=COLORS["gold"], edgecolors="white")
    axes[1].set(title="Observed division geometry", xlabel="Max parent → child distance (µm)", ylabel="Sister distance (µm)")
else: axes[1].text(.5,.5,"No divisions in profiled subset",ha="center",va="center")
for ax in axes: ax.grid(True)
plt.show()

if len(divisions):
    display(divisions[["parent_max_um","sister_um"]].describe(percentiles=[.5,.9,.95,.99]).T.style.format("{:.2f}"))
"""),
    markdown(r"""
### 9. Build balanced embryo-level folds

Random frame splits leak nearly identical neighboring frames. We assign whole embryos to folds, greedily balancing annotated nodes.
"""),
    code(r"""
if len(profiles):
    embryo_stats = profiles.groupby("embryo", as_index=False).agg(
        datasets=("dataset","nunique"), annotated_nodes=("annotated_nodes","sum"),
        edges=("edges","sum"), divisions=("divisions","sum"),
    ).sort_values(["annotated_nodes","divisions"], ascending=False)
    loads = [0] * min(N_FOLDS, len(embryo_stats)); assignments = []
    for row in embryo_stats.itertuples(index=False):
        fold = int(np.argmin(loads)); loads[fold] += int(row.annotated_nodes)
        assignments.append({"embryo": row.embryo, "fold": fold})
    assignments = pd.DataFrame(assignments)
    fold_table = embryo_stats.merge(assignments, on="embryo").groupby("fold", as_index=False).sum(numeric_only=True)
    display(fold_table.style.format("{:,.0f}").hide(axis="index"))
    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    ax.bar(fold_table.fold.astype(str), fold_table.annotated_nodes, color=COLORS["blue"], edgecolor="white")
    ax.set(title="Grouped-fold balance", xlabel="Validation fold", ylabel="Annotated nodes")
    ax.grid(axis="y"); plt.show()
    split_path = WORK_DIR / "research_embryo_folds.csv"
    assignments.to_csv(split_path, index=False)
    print("Saved:", split_path)
else: print("Fold construction skipped: no graph profiles.")
"""),
    markdown("## Takeaways"),
    code(r"""
findings = []
if len(catalog): findings.append(f"Cataloged {len(catalog):,} volumes across {catalog.embryo.nunique():,} embryo identifiers.")
if len(profiles): findings.append(f"Profiled {len(profiles):,} labeled graphs with {int(profiles.divisions.sum()):,} observed division events.")
if len(distances): findings.append(f"Observed motion p95 is {distances.quantile(.95):.2f} µm; compare this with every linking-radius proposal.")
if len(divisions): findings.append(f"Division parent-distance p95 is {divisions.parent_max_um.quantile(.95):.2f} µm and sister-distance p95 is {divisions.sister_um.quantile(.95):.2f} µm.")
if len(shift):
    strongest = shift.iloc[shift.median_gap_in_pooled_sd.abs().argmax()]
    findings.append(f"Largest simple train/test median shift is {strongest.feature}: {strongest.median_gap_in_pooled_sd:+.2f} pooled SD.")

items = "".join(f"<li style='margin:8px 0'>{text}</li>" for text in findings)
display(HTML(f"<div style='padding:22px 26px;border-left:5px solid #176B87;background:#F2F8FA;border-radius:12px'><b>Executed evidence</b><ul>{items}</ul></div>"))
"""),
    markdown(r"""
### How to use this atlas

1. Run once in `FAST_MODE` to validate the environment and inspect the regimes.
2. Switch to `FAST_MODE=False` for the durable full catalog.
3. Use `research_embryo_folds.csv` for every threshold, repair, and model comparison.
4. Cache raw model predictions once per checkpoint, then evaluate post-processing variants without rerunning the UNet.
5. Promote an idea only when it improves the mean fold score, does not collapse the worst embryo, and remains stable under intensity/noise perturbations.

### Next experiments

- detection-threshold calibration by imaging regime;
- confidence-aware short-track filtering;
- forward/backward edge consistency;
- division-specific evaluation on embryos with sufficient events;
- raw graph → ILP → repair ablation waterfall.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "None",
        "kaggle": {"accelerator": None, "dataSources": [], "dockerImageVersionId": None},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print(f"Wrote {OUTPUT}")
