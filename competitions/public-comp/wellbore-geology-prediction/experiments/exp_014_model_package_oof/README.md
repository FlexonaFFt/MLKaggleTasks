# EXP-014 — Model-package OOF audit

Executed local audit of `pilkwang/rogii-model-package` OOF artifacts.

Decision: retain the package only as a candidate bounded residual. Do not raise the Kaggle p95 guard or choose a blend weight until production-anchor OOF predictions are aligned to the same IDs.

Large downloaded OOF files live under the ignored `data/` directory.
