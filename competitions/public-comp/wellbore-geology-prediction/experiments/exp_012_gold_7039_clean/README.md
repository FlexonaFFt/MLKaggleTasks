# EXP-012 — Gold 7.039 clean rerun

Source: `/Users/flexonafft/Downloads/best-score.ipynb`.

Purpose: reproduce the reported 7.039 code-submission pipeline without changing its prediction logic.

Changes:

- removed network `pip install` calls that always fail with Kaggle internet disabled;
- install only the attached pure-Python `koolbox` wheel when the package is absent;
- do not install unrelated or ABI-incompatible wheels;
- append a read-only final audit for schema, row order, finite values, and hashes.

The source of every prediction cell is unchanged. `build_manifest.json` records the source hash and exact changed cell indices.

Required Kaggle inputs are declared in `kernel-metadata.json`. Kaggle must produce a single `/kaggle/working/submission.csv`.
