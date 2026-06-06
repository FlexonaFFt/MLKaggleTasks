# Predicting Loan Payback

Binary classification pipeline for predicting whether a loan is paid back.

## Status

Reproducible Python pipeline with CLI-style train and predict commands.

## What to Review

- `pipeline.py` - train/predict entry point.
- `catboost_model.py` - CatBoost model wrapper.
- `artifacts/metadata.json` - saved metadata for the recorded model.

## Usage

```bash
python3 pipeline.py train
python3 pipeline.py predict
```

Run commands from this folder. Expected local data is stored in `datasets/` and
is ignored by git.
