# EXP-013 — Tail-guarded model-package residual

Submission experiment on top of the confirmed anchor.

- model-package correction is required and cannot be disabled by a later A/B cell;
- maximum residual weight: `0.01`;
- disagreement scale: `6 ft`;
- global rollback when package/base disagreement p95 exceeds `25 ft`;
- execution fails before submission audit if the package stage did not actually run.

Kaggle title: `ROGII | EXP-013 Tail-Guarded Model Residual`.
