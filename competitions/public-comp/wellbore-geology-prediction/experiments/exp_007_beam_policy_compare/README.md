# EXP-007 — beam policy comparison

Full local run: 773 wells, cuts 20%, 25%, 33%, 2,319 pseudo-test cases, 273 seconds.

| Policy | Pooled RMSE | P90 | Applied cases |
|---|---:|---:|---:|
| v1 gate8 clip20 | 20.815831 | 25.893600 | 1,140 |
| gate8 clip15 | 20.822270 | 25.879238 | 1,140 |
| horizon 0.15 to 0.05 | 20.825069 | 25.947810 | 1,140 |
| gate6 clip20 | 20.826356 | 26.041820 | 936 |
| last value | 20.850403 | 26.267909 | 0 |

Decision: keep confirmed Kaggle 7.110 v1 policy. Do not spend hidden-scoring submissions on the three tested alternatives.

Caveat: local anchor is legal `last_value`, not hidden 7.110 trajectory. Use comparison for policy ranking, not absolute Kaggle score prediction.
