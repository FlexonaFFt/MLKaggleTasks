# Predicting Students Test Scores

Regression pipeline for predicting student exam scores.

## Status

Reproducible Python pipeline with separate data and trainer modules.

## What to Review

- `pipeline.py` - main feature engineering, validation, and training pipeline.
- `data.py` - dataset loading and preprocessing utilities.
- `trainers/` - LightGBM and CatBoost trainer implementations.

## Usage

```bash
python3 pipeline.py
```

Run the command from this folder. Expected local data is stored in `datasets/`
and is ignored by git.
