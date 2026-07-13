# EXP-010 — veto-only selector

Nested GroupKFold predictions reused from EXP-008/009. No beam recomputation.

| Policy | Pooled RMSE | P90 |
|---|---:|---:|
| v1 continuous cap | 20.814866 | 25.893600 |
| v1 | 20.815831 | 25.893600 |
| v1 double veto | 20.816103 | 25.893600 |
| v1 binary veto | 20.817285 | 25.893600 |
| v1 probability shrink | 20.827060 | 26.005982 |

Continuous cap improves v1 by only `0.000965 RMSE` (`0.0046%`). Too small for expensive hidden scoring and far below validation uncertainty. Decision: no Kaggle submission. Beam selector branch closed; retain confirmed v1 score 7.110.
