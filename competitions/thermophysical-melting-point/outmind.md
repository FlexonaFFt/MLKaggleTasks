# Leaky pipeline report (Thermophysical Melting Point)

## What this pipeline does (and where it leaks)
- **External join without cleanup**: External features are merged by raw `SMILES` without canonicalization and without removing overlaps with test data. This directly injects test-related information into train.
- **Global median imputation**: Missing values are filled using medians computed on the concatenation of train+test. This exposes test distribution to the training process.
- **Global scaling**: `StandardScaler` is fit on train+test combined. This leaks test statistics into train features.
- **KFold without grouping**: Standard `KFold` is used even though identical or near-duplicate `SMILES` can exist across folds, allowing the model to see near-duplicates during validation.
- **Synthetic leaky features**: If `external.csv` has only `SMILES`, a global frequency feature is built from train+test combined.

## Why these are harmful
- They yield an **artificially optimistic CV score**, which does not generalize to the public test set.
- They **contaminate the validation split** and hide real generalization errors.
- They **mislead model selection** and create false confidence in performance.

## How to do it correctly
- **Join external features only from train** or ensure the external source does not contain the test set.
- **Canonicalize SMILES** before merging to reduce duplicate leakage and stabilize joins.
- **Impute using train only**, and apply those statistics to validation/test splits.
- **Fit scalers within each fold** (or use sklearn Pipelines) to avoid leakage.
- **Use GroupKFold** by `SMILES` (or another grouping key) to prevent duplicates across folds.
