# Wellbore geology prediction

Kaggle competition workspace. The source data stays in `datasets/` (local and
ignored); generated experiment outputs stay next to the notebook that made
them.

## Competition status — 2026-07-30

- Status: active; deadline is **2026-08-05 23:59**. Kaggle reports **5,964 teams**.
- The visible `test/` directory contains authoring examples only. Kaggle replaces
  it with the hidden test set when a submission runs; do not use same-ID
  train/test matches from the visible files in model logic or validation.
  [Official clarification](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/729837#3503960).
- The organizer previously excluded one private-test outlier during a rescore;
  this does not change the public leaderboard.
  [Official notice](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707695#3471420).

### What matters now

- Treat public-LB changes smaller than the pipeline's rerun variation as noise;
  measure that variation with byte-identical reruns before choosing a branch.
  [Community analysis](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728477).
- Prefer target-free, well-grouped validation. Recent public notebooks are useful
  implementation references, not evidence that their public score will transfer
  to hidden wells.
- The two fresh public notebook variants checked on 2026-07-30 both retain a
  same-well contact override. Keep that branch disabled for hidden-test runs.

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
