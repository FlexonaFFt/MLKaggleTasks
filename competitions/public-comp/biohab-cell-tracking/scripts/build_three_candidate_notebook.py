from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "biohub_0902_anchor_global_shift.ipynb"
TARGET = ROOT / "notebooks" / "biohub_patched_three_candidate_selector.ipynb"
EVALUATOR_TARGET = ROOT / "notebooks" / "biohub_patched_short_candidate_evaluator.ipynb"


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    assert count == 1, f"Expected one replacement, found {count}: {old[:80]!r}"
    return text.replace(old, new)


notebook = json.loads(SOURCE.read_text())
cells = notebook["cells"]
assert len(cells) == 16

set_source(cells[1], """# Biohub Patched Metric: Clean 3-Way Selector

This notebook keeps the scored public V31 pipeline and produces three clean candidates:

1. `v31_standard` — the 0.902 control;
2. `v31_global_shift` — the same graph with conservative global-motion relinking;
3. `v31_ilp14` — the public high-continuity ILP weights (`appearance=0.0`, `disappearance=1.4`).

The patched division-metric exploit is intentionally absent. All candidate CSV files are
kept in Kaggle Output. `submission.csv` is an exact copy of the best **validated** candidate.
""")

set_source(cells[2], """## Pipeline

1. Run D4-TTA detector and transformer edge scorer for two ILP configurations.
2. Build standard and global-shift candidates from the V31 prediction graph.
3. Build the high-continuity candidate from the ILP-1.4 prediction graph.
4. Read patched holdout scores from `biohub_candidate_scores.json` and promote the winner.

Test labels are unavailable, so graph size or test geometry is never used as a fake score.
Without a score artifact, the notebook safely falls back to the known 0.902 control.
""")

set_source(cells[6], """## Candidate selection contract

Attach a small dataset containing `biohub_candidate_scores.json`, produced by the patched
local evaluator. Expected schema:

```json
{
  "metric_version": "patched-2026-07-17",
  "scores": {
    "v31_standard": 0.9020,
    "v31_global_shift": 0.9031,
    "v31_ilp14": 0.9052
  }
}
```

Only finite scores for known candidates are accepted. `BIOHUB_SELECTED_VARIANT` can override
the file for an explicit reproducible Kaggle run.
""")

set_source(cells[7], """CANDIDATE_NAMES = ("v31_standard", "v31_global_shift", "v31_ilp14")
CANDIDATE_SCORE_FILENAME = "biohub_candidate_scores.json"
SELECTED_VARIANT_OVERRIDE = os.environ.get("BIOHUB_SELECTED_VARIANT", "").strip()

ILP14_APPEARANCE_WEIGHT = 0.0
ILP14_DISAPPEARANCE_WEIGHT = 1.4

if SELECTED_VARIANT_OVERRIDE and SELECTED_VARIANT_OVERRIDE not in CANDIDATE_NAMES:
    raise ValueError(
        f"BIOHUB_SELECTED_VARIANT must be one of {CANDIDATE_NAMES}, "
        f"got {SELECTED_VARIANT_OVERRIDE!r}"
    )
""")

setup = source(cells[5])
setup = replace_once(
    setup,
    '''SUBMISSION_PATH = WORKING_DIR / "submission.csv"\nSUBMISSION_CONTROL_PATH = WORKING_DIR / "submission_control.csv"\nSUBMISSION_GLOBAL_SHIFT_PATH = WORKING_DIR / "submission_global_shift.csv"\nSUBMISSION_METRIC_HACK_PATH = WORKING_DIR / "submission_metric_hack.csv"\nRUN_STATS_PATH = WORKING_DIR / "run_stats.csv"\nRUN_STATS_GLOBAL_SHIFT_PATH = WORKING_DIR / "run_stats_global_shift.csv"''',
    '''SUBMISSION_PATH = WORKING_DIR / "submission.csv"\nSUBMISSION_STANDARD_PATH = WORKING_DIR / "submission_standard.csv"\nSUBMISSION_CONTROL_PATH = WORKING_DIR / "submission_control.csv"\nSUBMISSION_GLOBAL_SHIFT_PATH = WORKING_DIR / "submission_global_shift.csv"\nSUBMISSION_ILP14_PATH = WORKING_DIR / "submission_ilp14.csv"\nCANDIDATE_SELECTION_PATH = WORKING_DIR / "candidate_selection.json"\nRUN_STATS_PATH = WORKING_DIR / "run_stats.csv"\nRUN_STATS_GLOBAL_SHIFT_PATH = WORKING_DIR / "run_stats_global_shift.csv"\nRUN_STATS_ILP14_PATH = WORKING_DIR / "run_stats_ilp14.csv"''',
)
setup = replace_once(
    setup,
    'EXPERIMENT_TAG = "v31_0902_anchor_plus_global_shift"',
    'EXPERIMENT_TAG = "patched_clean_three_candidate_selector"',
)
set_source(cells[5], setup)

prediction = source(cells[11])
start = prediction.index("predict_cmd = [")
end_marker = 'print(f"Prediction completed in {predict_seconds / 60:.2f} minutes")'
end = prediction.index(end_marker) + len(end_marker)
new_prediction = '''def build_predict_cmd(appearance_weight: float, disappearance_weight: float) -> list[str]:
    command = [
        sys.executable,
        "scripts/predict_unet_transformer.py",
        "--data-dir", str(TEST_DIR),
        "--splits", str(splits_path.name),
        "--split", "0",
        "--weights", WEIGHTS_RELATIVE,
        "--unet-batch-size", str(UNET_BATCH_SIZE),
        "--det-threshold", str(DET_THRESHOLD),
        "--ilp-edge-weight", str(ILP_EDGE_WEIGHT),
        "--ilp-appearance-weight", str(appearance_weight),
        "--ilp-disappearance-weight", str(disappearance_weight),
        "--ilp-division-weight", str(ILP_DIVISION_WEIGHT),
    ]
    if USE_ILP:
        command.append("--use-ilp")
    if SLICE:
        command.extend(["--slice", SLICE])
    return command


prediction_runs = [
    ("biohub_v31_standard", ILP_APPEARANCE_WEIGHT, ILP_DISAPPEARANCE_WEIGHT),
    ("biohub_v31_ilp14", ILP14_APPEARANCE_WEIGHT, ILP14_DISAPPEARANCE_WEIGHT),
]
run_started = time.time()
processes = []
gpu_count = _torch_runtime_check.cuda.device_count()

for run_index, (run_name, appearance_weight, disappearance_weight) in enumerate(prediction_runs):
    command = build_predict_cmd(appearance_weight, disappearance_weight)
    run_env = {
        **os.environ,
        "USER": run_name,
        "USERNAME": run_name,
        "PYTHONPATH": "src",
    }
    if gpu_count >= len(prediction_runs):
        run_env["CUDA_VISIBLE_DEVICES"] = str(run_index)
        print(f"GPU {run_index}: {' '.join(command)}")
        processes.append((run_name, command, subprocess.Popen(command, cwd=REPO_DIR, env=run_env)))
    else:
        print(" ".join(command))
        subprocess.run(command, cwd=REPO_DIR, env=run_env, check=True)

for run_name, command, process in processes:
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    print(f"Finished {run_name}")

predict_seconds = time.time() - run_started
print(f"Two prediction variants completed in {predict_seconds / 60:.2f} wall-clock minutes")'''
prediction = prediction[:start] + new_prediction + prediction[end:]
set_source(cells[11], prediction)

build = source(cells[13])
build = replace_once(
    build,
    """    added: list[dict[str, object]] = []
    used_targets: set[int] = set()
""",
    """    added: list[dict[str, object]] = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()
""",
)
build = replace_once(
    build,
    """            if candidate_id in used_targets or candidate_id in incoming:
                continue
""",
    """            if source_id in used_sources or candidate_id in used_targets or candidate_id in incoming:
                continue
""",
)
build = replace_once(
    build,
    """            used_targets.add(candidate_id)
            added_this_frame += 1
""",
    """            used_sources.add(source_id)
            used_targets.add(candidate_id)
            added_this_frame += 1
""",
)
old_tail_start = build.index("final_predictions_dir =")
new_tail = '''def prediction_geffs(run_name: str) -> list[Path]:
    prediction_dir = REPO_DIR / "predictions" / run_name / METHOD / "split_0"
    paths = sorted(prediction_dir.glob("*.geff"))
    expected = set(test_stems)
    found = {path.stem for path in paths}
    if found != expected:
        raise RuntimeError({
            "run": run_name,
            "expected": len(expected),
            "found": len(found),
            "missing": sorted(expected - found)[:10],
            "extra": sorted(found - expected)[:10],
        })
    return paths


def build_submission(
    output_path: Path,
    stats_path: Path,
    geff_paths: list[Path],
    variant: str,
    use_global_shift: bool,
) -> pd.DataFrame:
    stats_rows: list[dict[str, object]] = []
    seen_datasets: set[str] = set()
    row_id = 0
    total_nodes = 0
    total_edges = 0

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for geff_path in geff_paths:
            dataset = geff_path.stem
            seen_datasets.add(dataset)
            graph = graph_from_geff(geff_path)

            nodes_by_id: dict[int, dict[str, object]] = {}
            for row in graph.node_attrs().iter_rows(named=True):
                node_id = int(row["node_id"])
                nodes_by_id[node_id] = {
                    "node_id": node_id,
                    "t": int(row["t"]),
                    "z": float(row["z"]),
                    "y": float(row["y"]),
                    "x": float(row["x"]),
                }

            raw_edges: list[dict[str, object]] = []
            for row in graph.edge_attrs().iter_rows(named=True):
                edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
                raw_edges.append({
                    "source_id": int(row["source_id"]),
                    "target_id": int(row["target_id"]),
                    "edge_prob": None if edge_prob is None else float(edge_prob),
                })

            raw_node_count = len(nodes_by_id)
            nodes_by_id, edges, filter_stats = filter_output_graph(
                nodes_by_id,
                raw_edges,
                dataset=dataset,
                deepcenter_bundle=DEEPCENTER_VETO_DETECTOR,
                use_global_shift=use_global_shift,
            )
            if not nodes_by_id:
                raise AssertionError(f"{dataset}: post-processing removed every node")

            for node_id in sorted(nodes_by_id):
                node = nodes_by_id[node_id]
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": max(0, int(round(float(node["z"])))),
                    "y": max(0, int(round(float(node["y"])))),
                    "x": max(0, int(round(float(node["x"])))),
                    "source_id": -1,
                    "target_id": -1,
                })
                row_id += 1

            division_sources: dict[int, int] = {}
            for edge in edges:
                source_id = int(edge["source_id"])
                target_id = int(edge["target_id"])
                if source_id not in nodes_by_id or target_id not in nodes_by_id:
                    raise AssertionError(f"{dataset}: dangling edge after filtering")
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": source_id,
                    "target_id": target_id,
                })
                row_id += 1
                division_sources[source_id] = division_sources.get(source_id, 0) + 1
                if division_sources[source_id] > 2:
                    raise AssertionError(f"{dataset}: node {source_id} has more than two children")

            node_count = len(nodes_by_id)
            edge_count = len(edges)
            total_nodes += node_count
            total_edges += edge_count
            stats_rows.append({
                "dataset": dataset,
                "raw_nodes": raw_node_count,
                "nodes": node_count,
                "raw_edges": filter_stats["raw_edges"],
                "edges": edge_count,
                "division_like_sources": sum(1 for count in division_sources.values() if count >= 2),
                "edge_to_node_ratio": edge_count / max(node_count, 1),
                "gap_added_nodes_frac": filter_stats.get("gap_added_nodes", 0) / max(raw_node_count, 1),
                **filter_stats,
            })

    expected_datasets = set(test_stems)
    if seen_datasets != expected_datasets:
        raise AssertionError({
            "missing": sorted(expected_datasets - seen_datasets)[:10],
            "extra": sorted(seen_datasets - expected_datasets)[:10],
        })
    assert row_id == total_nodes + total_edges, "Internal row counter mismatch"
    assert total_nodes > 0, "No node rows produced"
    assert output_path.open().readline().strip().split(",") == CSV_COLUMNS

    stats = pd.DataFrame(stats_rows).sort_values("dataset").reset_index(drop=True)
    stats["predict_minutes_total"] = predict_seconds / 60.0
    stats["experiment_tag"] = EXPERIMENT_TAG + "_" + variant
    stats.to_csv(stats_path, index=False)
    print(f"Wrote {output_path}: {total_nodes:,} nodes, {total_edges:,} edges")
    return stats


standard_geffs = prediction_geffs("biohub_v31_standard")
ilp14_geffs = prediction_geffs("biohub_v31_ilp14")

stats = build_submission(
    SUBMISSION_STANDARD_PATH, RUN_STATS_PATH, standard_geffs, "v31_standard", False
)
global_shift_stats = build_submission(
    SUBMISSION_GLOBAL_SHIFT_PATH,
    RUN_STATS_GLOBAL_SHIFT_PATH,
    standard_geffs,
    "v31_global_shift",
    True,
)
ilp14_stats = build_submission(
    SUBMISSION_ILP14_PATH, RUN_STATS_ILP14_PATH, ilp14_geffs, "v31_ilp14", False
)
shutil.copy2(SUBMISSION_STANDARD_PATH, SUBMISSION_CONTROL_PATH)

candidate_paths = {
    "v31_standard": SUBMISSION_STANDARD_PATH,
    "v31_global_shift": SUBMISSION_GLOBAL_SHIFT_PATH,
    "v31_ilp14": SUBMISSION_ILP14_PATH,
}
candidate_stats = {
    "v31_standard": stats,
    "v31_global_shift": global_shift_stats,
    "v31_ilp14": ilp14_stats,
}
comparison = pd.DataFrame([
    {
        "variant": variant,
        "nodes": int(frame["nodes"].sum()),
        "edges": int(frame["edges"].sum()),
        "shift_frames": int(frame.get("global_shift_frames", pd.Series([0])).sum()),
        "shift_edges": int(frame.get("global_shift_edges", pd.Series([0])).sum()),
    }
    for variant, frame in candidate_stats.items()
])


def load_candidate_scores() -> tuple[dict[str, float], str | None, str | None]:
    explicit = os.environ.get("BIOHUB_CANDIDATE_SCORES", "").strip()
    score_path = Path(explicit) if explicit else None
    if score_path is None:
        matches = sorted(Path("/kaggle/input").glob(f"**/{CANDIDATE_SCORE_FILENAME}")) \
            if Path("/kaggle/input").exists() else []
        score_path = matches[0] if matches else None
    if score_path is None or not score_path.exists():
        return {}, None, None

    payload = json.loads(score_path.read_text())
    if payload.get("metric_version") != "patched-2026-07-17":
        raise ValueError(f"Unsupported metric_version in {score_path}: {payload.get('metric_version')!r}")
    scores = {
        name: float(value)
        for name, value in payload.get("scores", {}).items()
        if name in CANDIDATE_NAMES and math.isfinite(float(value))
    }
    return scores, str(score_path), payload.get("metric_version")


candidate_scores, score_source, metric_version = load_candidate_scores()
if SELECTED_VARIANT_OVERRIDE:
    selected_variant = SELECTED_VARIANT_OVERRIDE
    selection_reason = "explicit BIOHUB_SELECTED_VARIANT override"
elif candidate_scores:
    selected_variant = max(candidate_scores, key=candidate_scores.get)
    selection_reason = "highest patched holdout score"
else:
    selected_variant = "v31_standard"
    selection_reason = "safe fallback: no patched holdout score artifact attached"

shutil.copy2(candidate_paths[selected_variant], SUBMISSION_PATH)
assert SUBMISSION_PATH.read_bytes() == candidate_paths[selected_variant].read_bytes()

selection = {
    "selected_variant": selected_variant,
    "selection_reason": selection_reason,
    "metric_version": metric_version,
    "score_source": score_source,
    "scores": candidate_scores,
    "candidate_files": {name: str(path) for name, path in candidate_paths.items()},
}
CANDIDATE_SELECTION_PATH.write_text(json.dumps(selection, indent=2, sort_keys=True))

print("Selected:", selected_variant)
print("Reason:", selection_reason)
print("submission.csv:", SUBMISSION_PATH)
display(comparison)
display(pd.DataFrame([
    {"variant": name, "patched_holdout_score": candidate_scores.get(name)}
    for name in CANDIDATE_NAMES
]))
display(pd.read_csv(SUBMISSION_PATH, nrows=8))
'''
set_source(cells[13], build[:old_tail_start] + new_tail)

set_source(cells[14], """## Outputs

The notebook always preserves all three candidates. Kaggle submits only `submission.csv`,
which is byte-for-byte identical to the selected candidate. `candidate_selection.json`
records the score source, metric version, winner, and fallback reason for auditability.
""")

set_source(cells[15], '''summary_columns = [
    "dataset", "nodes", "edges", "safe_divisions_added",
    "global_shift_frames", "global_shift_edges",
    "global_shift_magnitude_max_um",
]
summary_columns = [column for column in summary_columns if column in global_shift_stats.columns]
print("===== CLEAN THREE-CANDIDATE COMPARISON =====")
print(comparison.to_string(index=False))
print("\\nGlobal-shift activation by dataset:")
print(global_shift_stats[summary_columns].to_string(index=False))
print("\\nSelection:")
print(json.dumps(selection, indent=2, sort_keys=True))
print("\\nOutput files:")
for output_path in [
    SUBMISSION_PATH,
    SUBMISSION_STANDARD_PATH,
    SUBMISSION_GLOBAL_SHIFT_PATH,
    SUBMISSION_ILP14_PATH,
    CANDIDATE_SELECTION_PATH,
]:
    print(output_path)
''')

for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

assert "metric_hack" not in "\n".join(source(cell).lower() for cell in cells)
assert "submission_ilp14.csv" in source(cells[5])
assert "highest patched holdout score" in source(cells[13])

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print(TARGET)


evaluator = deepcopy(notebook)
evaluator_cells = evaluator["cells"]
set_source(evaluator_cells[1], """# Biohub Patched Metric: Short 3-Way Evaluator

Fast candidate comparison on one labeled training video. It runs the same D4 detector,
two ILP configurations, and three graph post-processing branches as the production notebook,
then scores every candidate with the official patched metric from commit `075fc5f`.

**Caveat:** this is an operational short run, not an unbiased generalization estimate. The
public checkpoint may have seen this video. Use its result to reject broken candidates and
choose the next leaderboard probe, not as a final validation claim.
""")

evaluator_config = source(evaluator_cells[4])
evaluator_config = replace_once(
    evaluator_config,
    "import os\n",
    '''import os

EVALUATION_DATASET = "44b6_0113de3b"
os.environ["BIOHUB_TEST_DIR"] = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"
os.environ["BIOHUB_EVALUATION_DATASET"] = EVALUATION_DATASET
''',
)
set_source(evaluator_cells[4], evaluator_config)

evaluator_prediction = source(evaluator_cells[11])
evaluator_prediction = replace_once(
    evaluator_prediction,
    '''    stems = sorted(path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"No test .zarr files found in {TEST_DIR}")
    return stems
''',
    '''    stems = sorted(path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith(".zarr"))
    selected = os.environ.get("BIOHUB_EVALUATION_DATASET", "").strip()
    if selected:
        stems = [stem for stem in stems if stem == selected]
    if not stems:
        raise FileNotFoundError(f"Evaluation dataset {selected!r} not found in {TEST_DIR}")
    return stems
''',
)
set_source(evaluator_cells[11], evaluator_prediction)

evaluator_cells.extend([
    {
        "cell_type": "markdown",
        "id": "patched-metric-evaluation",
        "metadata": {},
        "source": [
            "## Patched holdout scores\n",
            "\n",
            "Each candidate CSV is converted back to GEFF and evaluated against the paired "
            "training annotation. The resulting JSON is the selector input for the production notebook.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "score-clean-candidates",
        "metadata": {},
        "outputs": [],
        "source": '''PATCH_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
PATCH_METRIC_VERSION = "patched-2026-07-17"
SCORES_PATH = WORKING_DIR / CANDIDATE_SCORE_FILENAME

manifest_paths = sorted(Path("/kaggle/input").glob("**/PATCH_MANIFEST.json"))
patch_root = None
for manifest_path in manifest_paths:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("commit") == PATCH_COMMIT:
        patch_root = manifest_path.parent
        break
if patch_root is None:
    raise FileNotFoundError(
        "Attach flexonafft/biohub-patched-metric-v1; matching PATCH_MANIFEST.json was not found"
    )

metric_package = REPO_DIR / "src" / "biohub_tracking"
for filename in ("metrics.py", "division_metrics.py"):
    shutil.copy2(patch_root / filename, metric_package / filename)
print("Patched metric source:", patch_root, PATCH_COMMIT)

csv_converter = WORKING_DIR / "csv_to_geffs_patched.py"
csv_converter.write_text(
    (patch_root / "csv_to_geffs.py").read_text().replace(
        "from tracking_cellmot.io import save_graph",
        "from biohub_tracking.io import save_graph",
    )
)

score_rows = []
patched_scores = {}
evaluation_env = {**os.environ, "PYTHONPATH": str(REPO_DIR / "src")}
if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))
from geff import GeffMetadata
from biohub_tracking.io import open_dataset
from biohub_tracking.metrics import (
    evaluate as compute_metric,
    node_recall,
    per_sample_metrics,
    summarise,
)

gt_path = TEST_DIR / f"{EVALUATION_DATASET}.geff"
gt_metadata = GeffMetadata.read(gt_path)
estimated_nodes = float((gt_metadata.extra or {}).get("estimated_number_of_nodes", float("nan")))

for variant, csv_path in candidate_paths.items():
    geff_dir = WORKING_DIR / "patched_eval" / variant
    subprocess.run(
        [
            sys.executable,
            str(csv_converter),
            "--csv", str(csv_path),
            "--out-dir", str(geff_dir),
        ],
        cwd=REPO_DIR,
        env=evaluation_env,
        check=True,
    )
    prediction = graph_from_geff(geff_dir / f"{EVALUATION_DATASET}.geff")
    ground_truth = open_dataset(TEST_DIR / EVALUATION_DATASET, require_tracks=True)
    result = compute_metric(prediction, ground_truth.tracks, scale=ground_truth.scale)
    recall = node_recall(prediction, ground_truth.tracks)
    summary = summarise([per_sample_metrics(result, estimated_nodes, recall)])
    parsed = {
        name: float(summary[name])
        for name in ("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard", "node_recall")
    }
    if not math.isfinite(parsed["score"]):
        raise AssertionError(f"Non-finite score for {variant}: {parsed}")
    print(variant, parsed)
    patched_scores[variant] = parsed["score"]
    score_rows.append({"variant": variant, **parsed})

selected_variant = max(patched_scores, key=patched_scores.get)
selection_reason = "highest patched short-holdout score"
shutil.copy2(candidate_paths[selected_variant], SUBMISSION_PATH)
assert SUBMISSION_PATH.read_bytes() == candidate_paths[selected_variant].read_bytes()

score_payload = {
    "metric_version": PATCH_METRIC_VERSION,
    "metric_commit": PATCH_COMMIT,
    "evaluation_dataset": EVALUATION_DATASET,
    "validation_scope": "single-video-short-run",
    "scores": patched_scores,
    "selected_variant": selected_variant,
}
SCORES_PATH.write_text(json.dumps(score_payload, indent=2, sort_keys=True))
selection.update({
    "selected_variant": selected_variant,
    "selection_reason": selection_reason,
    "metric_version": PATCH_METRIC_VERSION,
    "score_source": str(SCORES_PATH),
    "scores": patched_scores,
})
CANDIDATE_SELECTION_PATH.write_text(json.dumps(selection, indent=2, sort_keys=True))

score_table = pd.DataFrame(score_rows).sort_values("score", ascending=False).reset_index(drop=True)
display(score_table)
print("Selected:", selected_variant)
print("Saved selector scores:", SCORES_PATH)
print(json.dumps(score_payload, indent=2, sort_keys=True))
'''.splitlines(keepends=True),
    },
])

for cell in evaluator_cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

evaluator_code = "\n".join(source(cell) for cell in evaluator_cells)
assert "BIOHUB_EVALUATION_DATASET" in evaluator_code
assert "PATCH_COMMIT" in evaluator_code
assert "single-video-short-run" in evaluator_code

EVALUATOR_TARGET.write_text(json.dumps(evaluator, ensure_ascii=False, indent=1) + "\n")
print(EVALUATOR_TARGET)
