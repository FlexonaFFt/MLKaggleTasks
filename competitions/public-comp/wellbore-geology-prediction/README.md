# Wellbore geology prediction

Kaggle competition workspace. The source data stays in `datasets/` (local and
ignored); generated experiment outputs stay next to the notebook that made
them.

## Layout

- `notebooks/` — Kaggle-ready notebooks.
  - `kaggle/` — submitted and submission-ready Kaggle iterations.
  - `research/` — diagnostic and hypothesis-testing notebooks.
  - `private/` — local/private analysis.
  - `reports/` — review reports.
  - `public_pipeline.ipynb` and `kernel-metadata.json` remain at the root as
    the legacy pipeline/config pair.
- `experiments/` — archived local research runs. One directory is one
  experiment; do not mix files between runs.
- `analysis/` — project-wide summaries.
  - `data-quality/` — dataset profiling and schema checks.
  - `leaderboard/` — scores, evidence, priorities, and submission history.
  - `catalog/` — generated artifact catalogues.
  - `reports/` — generated HTML reports.

## Working rules

1. Start a new experiment in `experiments/<short-name>/`.
2. Keep its notebook, `kernel-metadata.json`, script, and outputs together.
3. Put a reusable submission notebook in `notebooks/`; do not copy the same
   notebook both beside and inside its experiment folder.
4. Keep `.venv/`, `datasets/`, `__pycache__/`, and `.DS_Store` local only.
