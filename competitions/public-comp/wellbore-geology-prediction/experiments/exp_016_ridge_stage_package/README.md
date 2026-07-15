# EXP-016 — Ridge-stage package blend

Submission notebook based on EXP-015 grouped meta-validation.

```text
ridge_augmented = 0.60 * ridge + 0.40 * package
SP45 = 0.30 * ridge_augmented + 0.70 * selector
```

Projection, learned-track blend, contact guard, and visible-prefix layers remain downstream. The package is not applied again after the final blend.

Expected effective package contribution before guards is roughly 6.6%.
