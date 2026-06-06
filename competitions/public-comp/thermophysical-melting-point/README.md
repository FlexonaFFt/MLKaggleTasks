# Thermophysical Melting Point

Regression pipeline for predicting thermophysical melting point.

## Status

Reproducible Python pipeline with modular preprocessing and model code.

## What to Review

- `pipeline.py` - main training and submission pipeline.
- `mlcore/` - data processing, submission, and model wrappers.
- `datascreen/eda.ipynb` - exploratory data analysis notebook.
- `outmind.md` - leakage analysis and cleanup notes.

## Usage

```bash
python3 pipeline.py
```

Run the command from this folder. Expected local data is stored in `datasets/`
and is ignored by git.
