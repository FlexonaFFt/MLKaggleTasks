#!/usr/bin/env python3
"""One-command Kaggle portfolio sync for this repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = ROOT / "scripts" / "fetch_kaggle_stats.py"
PROFILE_URL = "https://www.kaggle.com/flexonafft"


def python_executable() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def main() -> int:
    return subprocess.run(
        [
            python_executable(),
            str(FETCH_SCRIPT),
            "--profile",
            PROFILE_URL,
            "--discover-entered",
            "--update-readme",
        ],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
