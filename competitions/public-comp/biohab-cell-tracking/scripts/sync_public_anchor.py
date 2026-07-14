#!/usr/bin/env python3
"""Pin the public Biohub notebook and inject a reproducible experiment preset."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = "kaiwalyaatulraut/biohub-competition-solution"
API_URL = f"https://www.kaggle.com/api/v1/kernels/pull/{KERNEL}"
OUTPUT = ROOT / "notebooks" / "biohub_anchor.ipynb"
LOCK = ROOT / "public_anchor.lock.json"

PRESET = {
    "BIOHUB_AUTO_SELECT": "1",
    "BIOHUB_CALIBRATION_THRESHOLDS": "0.95,0.97,0.985",
    "BIOHUB_CALIBRATION_PER_EMBRYO": "1",
    "BIOHUB_DET_THRESHOLD": "0.97",
    "BIOHUB_OUTPUT_FILTER_SHORT_TRACKS": "1",
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": "6",
    "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": "1",
    "BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE": "1",
    "BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC": "0.10",
    "BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN": "4",
    "BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": "0.82",
    "BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM": "3.25",
    "BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC": "0.018",
    "BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS": "180",
    "BIOHUB_GAP_CLOSE_MAX_GAP": "2",
    "BIOHUB_OUTPUT_GAP2_RECOVERY": "0",
    "BIOHUB_SAFE_DIV_MAX_UM": "4.66",
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "7.05",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "7.65",
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.0076",
    "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.00375",
    "BIOHUB_USE_DEEPCENTER_VETO": "0",
    "BIOHUB_RUN_VISUAL_EDA": "0",
    "BIOHUB_RUN_OUTPUT_DIAGNOSTICS": "1",
}

AUTO_SELECTION_MARKER = "# AUTO_SELECTION: choose the detector threshold before hidden-test inference"


def auto_selection_cells() -> list[dict]:
    markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Automatic candidate selection\n",
            "\n",
            "The notebook scores detector thresholds on one short labeled video per embryo, "
            "selects the best local candidate, and runs hidden-test inference only once. "
            "The absolute calibration score may be optimistic because the public checkpoint's "
            "training split is unknown; use it to compare candidates, not as an unbiased CV score.\n",
        ],
    }
    code = f'''{AUTO_SELECTION_MARKER}
TRAIN_DIR = COMP_DIR / "train"
AUTO_SELECT = os.environ.get("BIOHUB_AUTO_SELECT", "1") != "0"
CALIBRATION_THRESHOLDS = tuple(
    float(value) for value in os.environ.get(
        "BIOHUB_CALIBRATION_THRESHOLDS", "0.95,0.97,0.985"
    ).split(",") if value.strip()
)
CALIBRATION_PER_EMBRYO = int(os.environ.get("BIOHUB_CALIBRATION_PER_EMBRYO", "1"))
CALIBRATION_RESULTS_PATH = WORKING_DIR / "calibration_results.csv"
SELECTED_CONFIG_PATH = WORKING_DIR / "selected_config.json"
FINAL_PREDICTION_USER = "biohub_submission"


def choose_best_candidate(rows, anchor_threshold=0.97):
    valid = [row for row in rows if row.get("score") is not None and math.isfinite(row["score"])]
    if not valid:
        return None
    return max(valid, key=lambda row: (row["score"], -abs(row["threshold"] - anchor_threshold)))


assert choose_best_candidate([
    {{"threshold": 0.95, "score": 0.8}},
    {{"threshold": 0.97, "score": 0.8}},
])["threshold"] == 0.97

calibration_rows = []
selected_candidate = None
if AUTO_SELECT and TRAIN_DIR.exists() and CALIBRATION_THRESHOLDS:
    labeled = []
    for zarr_path in sorted(TRAIN_DIR.glob("*.zarr")):
        if not (TRAIN_DIR / f"{{zarr_path.stem}}.geff").exists():
            continue
        try:
            shape = tuple(json.loads((zarr_path / "0" / "zarr.json").read_text())["shape"])
        except Exception:
            continue
        labeled.append((zarr_path.stem.split("_", 1)[0], int(np.prod(shape)), zarr_path.stem))

    calibration_names = []
    for embryo in sorted({{row[0] for row in labeled}}):
        candidates = sorted(row for row in labeled if row[0] == embryo)
        calibration_names.extend(row[2] for row in candidates[:CALIBRATION_PER_EMBRYO])

    if calibration_names:
        calibration_split = REPO_DIR / "auto_selection_split.json"
        calibration_split.write_text(json.dumps([
            {{"split": 0, "train": [], "test": calibration_names}}
        ], indent=2))
        print("Calibration videos:", calibration_names)

        for threshold in CALIBRATION_THRESHOLDS:
            candidate_user = f"calibration_{{str(threshold).replace('.', '_')}}"
            candidate_env = {{
                **os.environ,
                "USER": candidate_user,
                "USERNAME": candidate_user,
                "BIOHUB_DATA_DIR": str(TRAIN_DIR),
                "PYTHONPATH": "src",
            }}
            candidate_cmd = [
                sys.executable, "scripts/predict_unet_transformer.py",
                "--data-dir", str(TRAIN_DIR),
                "--splits", str(calibration_split),
                "--split", "0",
                "--weights", WEIGHTS_RELATIVE,
                "--unet-batch-size", str(UNET_BATCH_SIZE),
                "--det-threshold", str(threshold),
                "--ilp-edge-weight", str(ILP_EDGE_WEIGHT),
                "--ilp-appearance-weight", str(ILP_APPEARANCE_WEIGHT),
                "--ilp-disappearance-weight", str(ILP_DISAPPEARANCE_WEIGHT),
                "--ilp-division-weight", str(ILP_DIVISION_WEIGHT),
            ]
            if USE_ILP:
                candidate_cmd.append("--use-ilp")

            started = time.time()
            predicted = subprocess.run(
                candidate_cmd, cwd=REPO_DIR, env=candidate_env,
                text=True, capture_output=True,
            )
            row = {{
                "threshold": threshold,
                "score": None,
                "runtime_minutes": (time.time() - started) / 60.0,
                "status": "prediction_failed" if predicted.returncode else "evaluation_failed",
            }}
            if predicted.returncode == 0:
                evaluated = subprocess.run([
                    sys.executable, "scripts/evaluate.py",
                    "--method", METHOD, "--split", "0", "--max-distance", "7.0",
                ], cwd=REPO_DIR, env=candidate_env, text=True, capture_output=True)
                match = re.search(r"(?<![A-Za-z_])score=([0-9.eE+-]+|nan)", evaluated.stdout, re.I)
                if evaluated.returncode == 0 and match and match.group(1).lower() != "nan":
                    row["score"] = float(match.group(1))
                    row["status"] = "ok"
                else:
                    row["error"] = (evaluated.stderr or evaluated.stdout)[-1000:]
            else:
                row["error"] = predicted.stderr[-1000:]
            calibration_rows.append(row)
            print(row)

        selected_candidate = choose_best_candidate(calibration_rows)
    else:
        print("AUTO_SELECTION warning: no labeled calibration videos found")

if selected_candidate is not None:
    DET_THRESHOLD = float(selected_candidate["threshold"])
    selection_reason = "best local calibration score"
else:
    selection_reason = "anchor fallback; calibration unavailable or disabled"

EXPERIMENT_TAG = f"auto_selected_det_{{DET_THRESHOLD:g}}"
os.environ["BIOHUB_DET_THRESHOLD"] = str(DET_THRESHOLD)
CONFIG_DISPLAY["det_threshold"] = DET_THRESHOLD
pd.DataFrame(calibration_rows).to_csv(CALIBRATION_RESULTS_PATH, index=False)
SELECTED_CONFIG_PATH.write_text(json.dumps({{
    "det_threshold": DET_THRESHOLD,
    "reason": selection_reason,
    "candidate": selected_candidate,
    "calibration_videos": locals().get("calibration_names", []),
}}, indent=2))
print(f"Selected det_threshold={{DET_THRESHOLD}}: {{selection_reason}}")
print("Final hidden-test output will be:", SUBMISSION_PATH)
'''
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }
    return [markdown, code_cell]


def inject_auto_selection(notebook: dict) -> None:
    if any(AUTO_SELECTION_MARKER in "".join(cell.get("source", [])) for cell in notebook["cells"]):
        return
    split_marker = "def list_test_stems() -> list[str]:"
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") != "code" or split_marker not in source:
            continue
        before, after = source.split(split_marker, 1)
        cell["source"] = before.splitlines(keepends=True)
        tail = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": (split_marker + after).splitlines(keepends=True),
        }
        notebook["cells"][index + 1:index + 1] = [*auto_selection_cells(), tail]
        return
    raise RuntimeError("Could not find the pre-inference insertion point")


def isolate_final_prediction_output(notebook: dict) -> None:
    """Keep calibration graphs out of the final submission graph set."""
    old_run = 'subprocess.run(predict_cmd, cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)'
    new_run = '''final_prediction_env = {
    **os.environ,
    "USER": FINAL_PREDICTION_USER,
    "USERNAME": FINAL_PREDICTION_USER,
    "PYTHONPATH": "src",
}
subprocess.run(predict_cmd, cwd=REPO_DIR, env=final_prediction_env, check=True)'''
    old_glob = 'geffs = sorted((REPO_DIR / "predictions").glob(f"*/{METHOD}/split_0/*.geff"))'
    new_glob = '''final_predictions_dir = REPO_DIR / "predictions" / FINAL_PREDICTION_USER / METHOD / "split_0"
geffs = sorted(final_predictions_dir.glob("*.geff"))'''

    found_run = found_glob = False
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if old_run in source:
            source = source.replace(old_run, new_run, 1)
            found_run = True
        elif new_run in source:
            found_run = True
        if old_glob in source:
            source = source.replace(old_glob, new_glob, 1)
            found_glob = True
        elif new_glob in source:
            found_glob = True
        cell["source"] = source.splitlines(keepends=True)
    if not found_run or not found_glob:
        raise RuntimeError("Could not isolate final prediction output")


def fetch() -> tuple[dict, dict]:
    request = urllib.request.Request(API_URL, headers={"User-Agent": "biohub-anchor/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return json.loads(payload["blob"]["source"]), payload["metadata"]


def preset_cell(version: int) -> dict:
    lines = [
        "# Local reproducible preset; generated by scripts/sync_public_anchor.py\n",
        "import os\n",
        f"BIOHUB_UPSTREAM_KERNEL = {KERNEL!r}\n",
        f"BIOHUB_UPSTREAM_VERSION = {version}\n",
    ]
    lines.extend(f"os.environ[{key!r}] = {value!r}\n" for key, value in PRESET.items())
    lines.append("print('Biohub anchor preset loaded:', BIOHUB_UPSTREAM_VERSION)\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="accept a newer upstream version")
    args = parser.parse_args()

    notebook, metadata = fetch()
    version = int(metadata["currentVersionNumberNullable"])
    if LOCK.exists():
        locked = json.loads(LOCK.read_text())
        if locked["version"] != version and not args.update:
            raise SystemExit(
                f"Upstream changed from v{locked['version']} to v{version}; "
                "inspect it and rerun with --update."
            )

    # Upstream cell 0 defines its defaults. Place our overrides immediately
    # after it and before any imports/config values are consumed.
    notebook["cells"].insert(1, preset_cell(version))
    inject_auto_selection(notebook)
    isolate_final_prediction_output(notebook)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    LOCK.write_text(json.dumps({"kernel": KERNEL, "version": version, "api_url": API_URL}, indent=2) + "\n")
    print(f"Wrote {OUTPUT} from {KERNEL} v{version}")


if __name__ == "__main__":
    main()
