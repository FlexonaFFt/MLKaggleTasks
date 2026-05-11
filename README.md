# Kaggle ML Portfolio

This repository is a curated portfolio of Kaggle and ML competition solutions.
Each competition lives in its own folder with the final pipeline, notebooks,
reports, and submission artifacts kept close to the original solution.

The repository is organized for review: the top-level README gives a quick
overview, while each project README explains what to inspect first and how the
solution is structured.

## Repository Structure

```text
.
├── competitions/       # Kaggle competition solutions
├── practice/           # Non-Kaggle practice and course tasks
├── scripts/            # Repository maintenance scripts
└── assets/             # Generated charts and shared README assets
```

## Featured Solutions

| Project | Type | Main entry point | Notes |
| - | - | - | - |
| [Predicting Students Test Scores](competitions/predicting-students-test-scores/) | Regression | `pipeline.py` | Full Python pipeline with LightGBM/CatBoost trainers |
| [Thermophysical Melting Point](competitions/thermophysical-melting-point/) | Regression | `pipeline.py` | Modular feature engineering and CatBoost pipeline |
| [Customer Churn](competitions/customer-churn/) | Binary classification | `train_models.py` | Research report, EDA report, model results |
| [Predicting Loan Payback](competitions/predicting-loan-payback/) | Binary classification | `pipeline.py` | CLI-style train/predict workflow |
| [Astronomical Classification](competitions/astronomical-classification/) | Time-series classification | `pipeline.ipynb` | Multiple notebook experiments including transformer baseline |

## Competition Index

| Project | Status | Primary files |
| - | - | - |
| [Astronomical Classification](competitions/astronomical-classification/) | Experiments + notebook solution | `pipeline.ipynb`, `optimal/transformer.ipynb` |
| [Christmas Tree Packing Challenge](competitions/christmas-tree-packing-challenge/) | Starter notebook | `getting-started.ipynb` |
| [Credit Card Fraud Detection](competitions/credit-card-fraud-detection/) | Notebook solution | `fraud-detection.ipynb` |
| [Customer Churn](competitions/customer-churn/) | Pipeline + reports | `train_models.py`, `pipeline.ipynb` |
| [LLM Classification Finetuning](competitions/llm-classification-finetuning/) | Baseline pipeline | `pipeline.py`, `notebooks/solution.ipynb` |
| [Predicting Heart Disease](competitions/predicting-heart-disease/) | Baseline notebooks | `mybaseline.ipynb`, `cvbaseline.ipynb` |
| [Predicting Irrigation Need](competitions/predicting-irrigation-need/) | EDA + notebook solution | `irrigation_need_solution.ipynb` |
| [Predicting Loan Payback](competitions/predicting-loan-payback/) | Reproducible pipeline | `pipeline.py`, `catboost_model.py` |
| [Predicting Pit Stop](competitions/predicting-pit-stop/) | Ensemble notebooks | `oopnote.ipynb`, `ensemblenote.ipynb`, `blender.ipynb` |
| [Predicting Students Test Scores](competitions/predicting-students-test-scores/) | Reproducible pipeline | `pipeline.py`, `data.py`, `trainers/` |
| [Rental Product Recommendation](competitions/rental-product-recommendation/) | Preprocessing notebook | `pipeline.ipynb` |
| [Sentiment Tweet Analysis](competitions/sentiment-tweet-analysis/) | Notebook solution | `tweet-analysis.ipynb` |
| [Thermophysical Melting Point](competitions/thermophysical-melting-point/) | Reproducible pipeline | `pipeline.py`, `mlcore/` |

Practice tasks are kept separately under [practice/](practice/).

## Results Tracker

The chart and table below are generated from `scripts/competitions.csv` with:

```bash
python3 scripts/update_readme.py
```

Formula:

```text
percent_beaten = (total - rank) / total * 100
```

<!-- COMPETITION-TABLE:START -->
| # | Competition | Rank | Total | % beaten |
| - | - | - | - | - |
| 1 | Predicting Students Test Score | 624 | 1741 | 64.2% |
| 2 | Thermophysical Property: Melting Point | 309 | 903 | 65.8% |
| 3 | LLM Classification | 141 | 238 | 40.8% |
| 4 | Astronomical Classification | 282 | 810 | 65.2% |
| 5 | Predicting Heart Disease | 70 | 536 | 86.9% |
<!-- COMPETITION-TABLE:END -->

<!-- PROGRESS-CHART:START -->
<img src="assets/progress.svg" alt="Kaggle progress chart" width="100%" />
<p><sub>Last updated: 2026-05-11 12:40</sub></p>
<!-- PROGRESS-CHART:END -->

## Data Policy

Competition datasets are expected to live inside each project as `datasets/` or
`dataset/`, but raw data is not part of the portfolio surface. Most local data,
training logs, caches, and environment files are ignored via `.gitignore`.

Run scripts from inside the project folder unless a project README states
otherwise. This preserves the original Kaggle-style relative paths.
