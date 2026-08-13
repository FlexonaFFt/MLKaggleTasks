#!/usr/bin/env python3
"""Pin the public dual-seed Biohub ensemble and reject partial submissions."""

from __future__ import annotations

import argparse
import io
import json
import re
import tokenize
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = "hiranorm/new-lb-0-916-infer-ensemble-lf-exp002"
API_URL = f"https://www.kaggle.com/api/v1/kernels/pull/{KERNEL}"
OUTPUT = ROOT / "notebooks" / "biohub_dual_seed_ensemble.ipynb"
LOCK = ROOT / "dual_seed_ensemble.lock.json"
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def set_cell_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def validate_notebook(notebook: dict) -> None:
    source = "\n".join(cell_source(cell) for cell in notebook["cells"])
    required = (
        "biohub-temporal-unet3d-seed314159-v1",
        "biohub-tracking-support-pack-50ep-v1",
        "'det_threshold': 0.90",
        "predict_video(models, zp, mc)",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise RuntimeError(f"Unexpected upstream notebook; missing {missing}")


def strip_notebook_notes(notebook: dict) -> None:
    cells = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "markdown":
            continue
        source = cell_source(cell)
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        source = tokenize.untokenize(token for token in tokens if token.type != tokenize.COMMENT)
        set_cell_source(cell, CJK.sub("", source))
        cells.append(cell)
    notebook["cells"] = cells


def add_partial_submission_guard(notebook: dict) -> None:
    marker = "n_rows = write_submission(results, OUT)"
    guard = """if skipped:
    raise RuntimeError(f\"Refusing partial submission; failed datasets: {skipped}\")
if len(results) != len(zpaths):
    raise RuntimeError(f\"Refusing incomplete submission: {len(results)} / {len(zpaths)} datasets\")
\n"""
    for cell in notebook["cells"]:
        source = cell_source(cell)
        if marker not in source:
            continue
        if "Refusing partial submission" not in source:
            source = source.replace(marker, guard + marker, 1)
            set_cell_source(cell, source)
        return
    raise RuntimeError("Could not add the submission-completeness guard")


def fetch() -> tuple[dict, dict]:
    request = urllib.request.Request(API_URL, headers={"User-Agent": "biohub-dual-seed/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return json.loads(payload["blob"]["source"]), payload["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="accept a newer upstream version")
    args = parser.parse_args()

    notebook, metadata = fetch()
    version = int(metadata["currentVersionNumberNullable"])
    if LOCK.exists() and not args.update:
        locked = json.loads(LOCK.read_text())
        if locked["version"] != version:
            raise SystemExit(f"Upstream changed from v{locked['version']} to v{version}; inspect then use --update.")

    validate_notebook(notebook)
    strip_notebook_notes(notebook)
    add_partial_submission_guard(notebook)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    LOCK.write_text(json.dumps({"kernel": KERNEL, "version": version, "api_url": API_URL}, indent=2) + "\n")
    print(f"Wrote {OUTPUT} from {KERNEL} v{version}")


if __name__ == "__main__":
    main()
