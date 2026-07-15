# Biohub cell tracking

Reproducible Kaggle-ready anchor based on the public
`kaiwalyaatulraut/biohub-competition-solution` notebook. The public solution
uses a Temporal UNet3D detector, a learned edge transformer, ILP graph
selection, spatial TTA, and conservative graph repair.

## Quick start

```bash
python scripts/sync_public_anchor.py
python -m unittest discover -s tests -v
python scripts/validate_submission.py /path/to/submission.csv
```

Upload `notebooks/biohub_anchor.ipynb` to Kaggle and attach:

- competition data `biohub-cell-tracking-during-development`;
- dataset `pilkwang/biohub-tracking-support-pack-50ep-v1`.

The generated notebook is pinned in `public_anchor.lock.json`. Re-running the
sync command refuses a changed upstream version unless `--update` is supplied.
This prevents an unnoticed public-notebook update from changing an experiment.

## Initial experiment

The checked-in preset preserves the public high-scoring anchor:

- detection threshold: `0.97`;
- tracks shorter than 6 nodes are filtered;
- adaptive short-track rescue is disabled;
- learned-edge motion bonus is `1.0`;
- gap repair and conservative division recovery remain enabled;
- D4-style spatial TTA remains enabled.

The notebook writes both `submission.csv` and `submission_global_shift.csv`
from the same raw predictions. Only the second candidate compensates confident
frame-level global motion during relinking.

## Files

- `scripts/sync_public_anchor.py`: downloads and pins the public notebook,
  then injects our explicit preset as the first cell.
- `scripts/validate_submission.py`: validates schema and lineage invariants.
- `notebooks/biohub_anchor.ipynb`: generated runnable Kaggle notebook.
- `experiments.csv`: experiment ledger.

The original notebook and support artifact belong to their respective Kaggle
authors. Keep their attribution when publishing forks.
