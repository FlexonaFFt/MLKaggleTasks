from pathlib import Path

import nbformat as nbf


root = Path(__file__).resolve().parents[1]
source = root / "rogii_frozen_targetfree_hybrid_v1" / "rogii_frozen_targetfree_hybrid_v1.ipynb"
out = Path(__file__).with_name("rogii_frozen_confidence_gate_v1.ipynb")
nb = nbf.read(source, as_version=4)

nb.cells[0] = nbf.v4.new_markdown_cell(
    """# ROGII Frozen confidence gate v1

Diagnostic only. Tests whether Frozen residual should be suppressed on 20% of
high-risk wells using features available before labels. No submission is written."""
)
nb.cells.pop()  # Replace the previous global-blend diagnostic.
nb.cells.append(
    nbf.v4.new_code_cell(
        """# Conditional residual gate: fixed, label-free, and diagnostic only.
_gate_eval = _v5_eval.copy()
_gate_features = _v6_wells.set_index('well', drop=False)
for _gate_column in ('spatial_fold', 'spatial_cluster', 'regime', 'residual_rms', 'suffix_md_span', 'suffix_gr_std'):
    _gate_eval[_gate_column] = _gate_eval['well'].map(dict(zip(_gate_features['well'], _gate_features[_gate_column])))
assert _gate_eval.notna().all().all()

_gate_baseline = _gate_eval['baseline'].to_numpy(float)
_gate_frozen = _gate_eval['primary'].to_numpy(float)
_gate_residual = _gate_frozen - _gate_baseline

def _gate_metrics(prediction):
    squared = (_gate_eval['target'].to_numpy(float) - prediction) ** 2
    rmse = float(_v5_np.sqrt(squared.mean()))
    by_well = _gate_eval.assign(squared=squared).groupby('well')['squared'].agg(['sum', 'mean'])
    tail = float(by_well.nlargest(max(1, len(by_well) // 10), 'mean')['sum'].sum() / by_well['sum'].sum())
    def wins(column):
        candidate = _gate_eval.assign(squared=squared).groupby(column)['squared'].mean().pow(.5)
        frozen = _gate_eval.assign(squared=(_gate_eval['target'].to_numpy(float) - _gate_frozen) ** 2).groupby(column)['squared'].mean().pow(.5)
        return int((candidate < frozen).sum())
    return rmse, tail, wins('spatial_fold'), wins('spatial_cluster'), wins('regime')

_gate_frozen_rmse, _gate_frozen_tail, _, _, _ = _gate_metrics(_gate_frozen)
_gate_specs = [('frozen', None)]
for _gate_feature in ('residual_rms', 'suffix_md_span', 'suffix_gr_std'):
    _gate_specs.append((_gate_feature + '_top20', _gate_feature))

_gate_rows = []
for _gate_name, _gate_feature in _gate_specs:
    if _gate_feature is None:
        _gate_prediction = _gate_frozen
        _gateed_wells = 0
    else:
        _gate_threshold = float(_gate_features[_gate_feature].quantile(.80))
        _gateed = _gate_eval[_gate_feature].to_numpy(float) >= _gate_threshold
        _gate_prediction = _gate_baseline + _gate_residual * (~_gateed)
        _gateed_wells = int(_gate_features[_gate_features[_gate_feature] >= _gate_threshold].shape[0])
    _gate_rmse, _gate_tail, _gate_spatial, _gate_clusters, _gate_regimes = _gate_metrics(_gate_prediction)
    _gate_rows.append({
        'rule': _gate_name, 'gated_wells': _gateed_wells, 'pooled_rmse': _gate_rmse,
        'gain_vs_frozen_ft': _gate_frozen_rmse - _gate_rmse, 'worst10_sse_share': _gate_tail,
        'spatial_fold_wins': _gate_spatial, 'cluster_wins': _gate_clusters, 'regime_wins': _gate_regimes,
        'eligible': bool(_gate_name != 'frozen' and _gate_rmse <= _gate_frozen_rmse - .02 and _gate_tail <= _gate_frozen_tail and _gate_spatial >= 4 and _gate_clusters >= 8 and _gate_regimes >= 4),
    })

_gate_summary = _v5_pd.DataFrame(_gate_rows)
print(_gate_summary.to_string(index=False))
assert len(_gate_summary) == 4

# Do not publish training artifacts from this diagnostic run.
import shutil as _gate_shutil
for _gate_path in ('/kaggle/working/models', '/kaggle/working/catboost_info'):
    _gate_shutil.rmtree(_gate_path, ignore_errors=True)
"""
    )
)
nbf.write(nb, out)
print(out)
