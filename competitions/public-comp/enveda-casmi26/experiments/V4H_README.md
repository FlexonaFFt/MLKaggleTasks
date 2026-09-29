# CASMI26 v4h: two submissions

Source: [Ahmed Berat Özer, v4h, scriptVersionId 353548586](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4h-inference?scriptVersionId=353548586).
The exact downloaded notebook is preserved in `sources/casmi26_v4h_353548586.ipynb`, including its outputs and dataset-version IDs.

| Notebook | Kaggle title | Change to inference |
| --- | --- | --- |
| `casmi26_exp_g_v4h_reference.ipynb` | CASMI26: v4h Faithful Reference | None: original weights, generation, fusion and gates |
| `casmi26_exp_h_v4h_glacier_quarter.ipynb` | CASMI26: v4h Gentle GLACIER | GLACIER weight 0.5 → 0.25 only |

The experiment reduces GLACIER's influence in same-formula reranking. It preserves ICEBERG's weight of 0.5, TOPN=60, both runtime budgets, all candidates and the PubChem insertion gates. This is a hypothesis, not a validated improvement.

## Inputs for both notebooks

Attach the competition and **all seven** datasets below. Existing COCONUT/fingerprint inputs from earlier experiments do not provide the packaged v4h pool and engines.

1. [casmi26-v4b-models](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models)
2. [casmi26-v3-models](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v3-models)
3. [casmi26-v2-pool](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v2-pool)
4. [casmi26-pubchem-tier](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-pubchem-tier)
5. [casmi26-iceberg](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-iceberg)
6. [casmi26-glacier](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-glacier)
7. [rdkit 2026.3.3 wheel](https://www.kaggle.com/datasets/metric/rdkit-2026-3-3-wheel)

Original dataset-version IDs are retained in each notebook's Kaggle metadata. Check the Input panel after import: importing a local notebook may not apply every metadata field. The source's MANIFEST hash checks remain intact.

Enable **GPU T4 x2 or P100**, internet off. The source's saved example-data execution took approximately 66 minutes; hidden-test timing can differ. Use Save & Run All, then submit the successful notebook version to the code competition.

Both write `submission.csv`, `run_manifest.json` and `validation_report.json`. Inspect the report: `degraded` means the original fallback behaviour was used after a channel failure. `complete` means channels produced results, not that every candidate was covered. Coverage statistics and channel logs remain available.

## Validation

`python3 build_v4h_experiments.py` regenerates both notebooks from the preserved source and checks syntax plus computational AST equivalence. Embedded CORE/RUNNER code was also syntax-checked. All cells use triple-quoted comment blocks in code cells, including the introduction.

The new notebooks have not been executed with Kaggle inputs or scored. Their GPU execution and leaderboard comparison remain necessary. The source's successful example-data outputs establish source execution only; they do not establish a hidden-test score for either adaptation.
