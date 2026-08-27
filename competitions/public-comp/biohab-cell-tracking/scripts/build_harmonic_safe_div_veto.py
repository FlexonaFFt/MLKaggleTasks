#!/usr/bin/env python3
"""Build the one-change DeepCenter veto variant from the live Harmonic Fusion notebook."""

from __future__ import annotations

import io
import json
import re
import tokenize
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = "flexonafft/biohub-harmonic-fusion"
API_URL = f"https://www.kaggle.com/api/v1/kernels/pull/{KERNEL}"
OUTPUT = ROOT / "notebooks" / "biohub_harmonic_safe_div_veto.ipynb"
TITLE = "Biohub Harmonic Fusion | DeepCenter Safe-Div Veto"
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
OVERRIDE = """import os
os.environ[\"BIOHUB_DEEPCENTER_EXPECTED_EPOCH\"] = \"2\"
os.environ[\"BIOHUB_DEEPCENTER_CHECKPOINT\"] = \"/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt\"
os.environ[\"BIOHUB_DEEPCENTER_SAFE_DIV_VETO\"] = \"1\"
"""


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def clean(notebook: dict) -> None:
    cells = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "markdown":
            continue
        tokens = tokenize.generate_tokens(io.StringIO(source(cell)).readline)
        set_source(cell, CJK.sub("", tokenize.untokenize(t for t in tokens if t.type != tokenize.COMMENT)))
        cells.append(cell)
    notebook["cells"] = cells


def inject_override(notebook: dict) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if "GAP_CLOSE_MAX_GAP = int(" in source(cell):
            notebook["cells"].insert(
                index,
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": OVERRIDE.splitlines(keepends=True)},
            )
            return
    raise RuntimeError("Could not find the configuration-read cell")


def validate(notebook: dict) -> None:
    text = "\n".join(source(cell) for cell in notebook["cells"])
    for required in (
        'BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = "2"',
        'full_frame_center/best.pt',
        'BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"',
        "BIOHUB_BIDIRECTIONAL_FUSION_MODE",
    ):
        if required not in text:
            raise RuntimeError(f"Missing required setting: {required}")
    if CJK.search(text):
        raise RuntimeError("Generated notebook contains CJK characters")


def fetch() -> dict:
    request = urllib.request.Request(API_URL, headers={"User-Agent": "biohub-safe-div-veto/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(json.load(response)["blob"]["source"])


def main() -> None:
    notebook = fetch()
    clean(notebook)
    inject_override(notebook)
    notebook.setdefault("metadata", {})["title"] = TITLE
    validate(notebook)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
