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

`biohub_patched_three_candidate_selector.ipynb` writes three clean candidates:
V31 standard, global-shift relinking, and the ILP-1.4 continuity variant. It
promotes the highest patched holdout score to `submission.csv`; without a
`biohub_candidate_scores.json` artifact it safely falls back to V31 standard.

`biohub_dual_seed_ensemble.ipynb` is the next experiment: it averages two
independent model logits before point extraction, then runs one graph optimizer.
Build it with `python3 scripts/sync_dual_seed_ensemble.py` and attach the two
weight datasets plus the competition data. It refuses to write a partial submission.

`biohub_harmonic_safe_div_veto.ipynb` is a one-shot Harmonic Fusion variant:
it uses DeepCenter `best.pt` (epoch 2) and enables its safe-division veto.

## Files

- `scripts/sync_public_anchor.py`: downloads and pins the public notebook,
  then injects our explicit preset as the first cell.
- `scripts/validate_submission.py`: validates schema and lineage invariants.
- `scripts/build_three_candidate_notebook.py`: reproducibly builds the clean
  three-candidate Kaggle notebook from the scored V31 anchor.
- `scripts/sync_dual_seed_ensemble.py`: pins the public dual-seed ensemble and
  adds a submission-completeness guard.
- `scripts/build_harmonic_safe_div_veto.py`: builds the DeepCenter safe-division
  veto variant from the live Harmonic Fusion notebook.
- `scripts/sync_harmonic_fusion_v17.py`: pins `yunusgmsoy/kimi-notebook-v17`
  (harmonic bidirectional fusion over the 0.913 dual-seed baseline), verifies
  its checksum guards, and disables the held-out train validator for scored
  reruns (`BIOHUB_VALIDATOR_ENABLE=1` re-enables it).
- `notebooks/biohub_anchor.ipynb`: generated runnable Kaggle notebook.
- `notebooks/biohub_patched_three_candidate_selector.ipynb`: patched-metric
  production notebook and validated-candidate selector.
- `notebooks/biohub_harmonic_fusion_v17.ipynb`: generated all-in fusion
  notebook (pin: `harmonic_fusion_v17.lock.json`). Attach the competition data
  plus `pilkwang/biohub-tracking-support-pack-50ep-v1`,
  `pilkwang/biohub-temporal-unet3d-seed314159-v1`, and
  `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`; run on GPU T4 x2 with
  Internet off.
- `experiments.csv`: experiment ledger.

The original notebook and support artifact belong to their respective Kaggle
authors. Keep their attribution when publishing forks.
