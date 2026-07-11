# EXP-004 — Kaggle 7.148 audit

Source kernel: `flexonafft/working-note-try-7-100`, submission `54530991`, scored 2026-07-10.

Observed facts:

- Public score: `7.148`.
- Visible output: 14,151 rows, SHA-256 `fdf4a8175b6ec6a70c9b78fd6916ac3c317e43f7e9c08bbca87cd02314801ca9`.
- Same visible SHA appeared in earlier pipeline output associated with the 7.201 run.
- Therefore code-competition score depends on Kaggle hidden rerun behavior, not visible three-well CSV.
- Selected profile: `vp_balanced_final`.
- SP45/learned weight: `0.60 / 0.40`.
- Projection: degree `3`, blend `0.75`.
- Guarded contact override passed on all three visible wells, replacing all 14,151 visible prediction rows.
- Bimodal detector and model-package correction were disabled.
- Hidden improvement cannot be attributed from one submission because profile, projection, runtime threading, and calibration implementation changed together.

Next controlled code submissions:

1. 7.148 kernel unchanged, reproducibility check.
2. Projection degree 4 only.
3. `vp_conservative_final` only.
4. Bounded `last + 0.25 * beam correction` only, gated for low-prefix wells.

One changed factor per submission. Record kernel version, score, runtime, and visible SHA in `experiments/registry.csv`.
