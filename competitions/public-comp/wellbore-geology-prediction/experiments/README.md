# ROGII experiments

Each experiment has one immutable ID and its own folder:

```text
exp_NNN_short_name/
  notebook.ipynb
  results/          # generated metrics and predictions
```

Rules:

1. Never overwrite an experiment after using its results for a decision; create the next ID.
2. Record the purpose, status, headline metric, and conclusion in `registry.csv`.
3. Validation experiments never use public-test targets or leaderboard scores for model selection.
4. Submission notebooks stay in `notebooks/`; experiments may promote only validated changes there.

## Current state

- `exp_001_prefix_validation`: completed on 773 wells and four cuts. It established the metric pipeline and exposed that same-well formation surfaces make a plain prefix split unrealistically easy. Results are preserved, but its `0.005370` formation-shape RMSE is not a trustworthy hidden-test estimate.
- `exp_002_legal_schema_validation`: completed on 773 wells. Train-only formation columns were removed; `last_value` is the legal pooled anchor at `11.493684`. Local trend selection is unstable, while `z_residual_linear` becomes competitive only for short horizons.
- `exp_003_gr_alignment`: completed in one full notebook on 773 wells and seven cuts, including competition-like 20/25/33% prefixes. `last_beam_w25` improved all-cut pooled RMSE `18.047750` to `16.065684`; on low-prefix cuts it improved `20.850403` to `18.087659`. Standalone PF failed badly and is rejected.
- Next ID: `exp_004`. Attribute Kaggle 7.148 gain with controlled notebook profiles, then test bounded beam correction inside strongest 7.148 pipeline.

Run the first experiment from the repository root:

```bash
.venv/bin/jupyter nbconvert --execute --to notebook --inplace \
  competitions/public-comp/wellbore-geology-prediction/experiments/exp_001_prefix_validation/notebook.ipynb
```
