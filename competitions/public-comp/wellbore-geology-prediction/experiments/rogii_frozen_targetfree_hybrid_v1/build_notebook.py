from pathlib import Path

import nbformat as nbf


root = Path(__file__).resolve().parents[1]
source = root / "rogii_frozen_anchor_residual_v1" / "rogii_frozen_anchor_residual_v1.ipynb"
out = Path(__file__).with_name("rogii_frozen_targetfree_hybrid_v1.ipynb")
source_nb = nbf.read(source, as_version=4)

cells = [
    nbf.v4.new_markdown_cell(
        """# ROGII Frozen × Target-Free OOF hybrid v1

Diagnostic only. `target_free` is the frozen notebook's cross-fitted Ridge/PF
core; `frozen_v6` adds its bounded residual. This notebook tests fixed blends
on identical target-free OOF rows and spatial folds. It never creates a
submission."""
    ),
    source_nb.cells[2],
    *source_nb.cells[3:19],
    nbf.v4.new_code_cell(
        """from pathlib import Path
import pandas as pd

# V5 needs a file only to preserve its audit path; diagnostic mode never promotes it.
_hybrid_work = Path('/kaggle/working')
_hybrid_sample = pd.read_csv(CFG.dataset_path / 'sample_submission.csv')[['id']]
_hybrid_sample.assign(tvt=0.0).to_csv(_hybrid_work / 'v5_input.csv', index=False)
"""
    ),
]

v5 = source_nb.cells[51].copy()
v5.source = v5.source.replace("_V5_SUBMISSION = _V5_WORK / 'submission.csv'", "_V5_SUBMISSION = _V5_WORK / 'v5_input.csv'")
v5.source = v5.source.replace("if _v5_viable:", "if False:  # diagnostic only: never write a candidate")
cells.append(v5)
cells.append(source_nb.cells[53])
cells.append(
    nbf.v4.new_code_cell(
        """# Aligned fixed-weight hybrid: no learned selector, no test labels, no submission.
_HYBRID_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
_hybrid_rows = []
_hybrid_eval = _v5_eval.merge(_v6_wells[['well', 'spatial_fold']], on='well', how='left')
assert _hybrid_eval['spatial_fold'].notna().all()

def _hybrid_metrics(frame, prediction):
    squared = (frame['target'].to_numpy(float) - prediction) ** 2
    rmse = float(_v5_np.sqrt(squared.mean()))
    by_well = frame.assign(squared=squared).groupby('well')['squared'].agg(['sum', 'mean'])
    tail = float(by_well.nlargest(max(1, len(by_well) // 10), 'mean')['sum'].sum() / by_well['sum'].sum())
    spatial = frame.assign(squared=squared).groupby('spatial_fold')['squared'].mean().pow(.5)
    return rmse, tail, spatial

frozen = _hybrid_eval['primary'].to_numpy(float)
target_free = _hybrid_eval['baseline'].to_numpy(float)
frozen_rmse, frozen_tail, frozen_spatial = _hybrid_metrics(_hybrid_eval, frozen)

for frozen_weight in _HYBRID_WEIGHTS:
    prediction = frozen_weight * frozen + (1.0 - frozen_weight) * target_free
    rmse, tail, spatial = _hybrid_metrics(_hybrid_eval, prediction)
    _hybrid_rows.append({
        'frozen_weight': frozen_weight,
        'target_free_weight': 1.0 - frozen_weight,
        'pooled_rmse': rmse,
        'gain_vs_frozen_ft': frozen_rmse - rmse,
        'worst10_sse_share': tail,
        'spatial_fold_wins_vs_frozen': int((spatial < frozen_spatial).sum()),
        'eligible': bool(rmse <= frozen_rmse - 0.05 and tail <= frozen_tail and (spatial < frozen_spatial).sum() >= 4),
    })

_hybrid_summary = pd.DataFrame(_hybrid_rows)
_hybrid_summary.to_csv(_V5_WORK / 'hybrid_oof_summary.csv', index=False)
print(_hybrid_summary.to_string(index=False))
assert len(_hybrid_summary) == len(_HYBRID_WEIGHTS)
"""
    )
)

nb = nbf.v4.new_notebook()
nb.metadata = source_nb.metadata
nb.cells = cells
nbf.write(nb, out)
print(out)
