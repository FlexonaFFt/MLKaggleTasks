# EXP-008/009 — learned beam selectors

Nested 5-fold GroupKFold, 773 wells, cuts 20%, 25%, 33%.

| Policy | Pooled RMSE | P90 | Worst |
|---|---:|---:|---:|
| Binary gate | 20.485260 | 25.704593 | 343.905070 |
| Continuous weight | 20.542858 | 25.749110 | 344.350561 |
| v1 hand gate | 20.815831 | 25.893600 | 346.759022 |
| Anchor | 20.850403 | 26.267909 | 346.759022 |

Binary gate selected. Both models embedded as plain coefficients in Kaggle notebook. Selection is fixed from OOF results; hidden test labels never participate.
