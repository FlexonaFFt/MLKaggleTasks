"""Optional raw+.03*SC, original alpha.5 balance/gate, then shared protected PFN.10."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from claude_gate import gated_predict, load_gate_state
from protected_operator import protected
from source_balance import source_balance

here = Path(__file__).resolve().parent
from admission_preflight import validate
config = validate(here)
assert config['candidate_admitted'] and config['selected_by_root']
assert config['assets_public_audited'] and config['diagnostic_actual_PASS_bound']
assert config['raw_extension'] == 'smol_center_add' and config['coefficient'] == .03
assert config['alpha'] == .5 and config['protected_weight'] == .10 and config['post'] == 0
for name, digest in config['source_sha256'].items():
    assert hashlib.sha256((here / name).read_bytes()).hexdigest() == digest, name
cache = Path(os.environ['NTL_TABULAR_TEMP']).resolve()
assert cache.is_relative_to(Path('/kaggle/temp').resolve())


def load(name):
    with np.load(cache / name, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


observed, smol, pfn = (load(name) for name in
    ('baseline_observation.npz', 'smol_center_add.npz', 'tabpfn_proba.npz'))
raw, P, ids, labels = (observed[key] for key in ('raw', 'balanced', 'ids', 'labels'))
assert raw.dtype == P.dtype == np.float64 and raw.shape == P.shape == (len(ids), 18)
for values in (smol, pfn):
    np.testing.assert_array_equal(values['ids'], ids)
    np.testing.assert_array_equal(values['labels'], labels)
for key in ('pool_ids', 'recovered_group_ids', 'adaptive_mask'):
    np.testing.assert_array_equal(smol[key], observed[key])
# The extracted original source balance must reproduce the observed baseline bit-for-bit.
np.testing.assert_array_equal(source_balance(raw), P)
gate = load_gate_state(cache, P, ids, labels)
baseline = gated_predict(P, labels, **gate)
np.testing.assert_array_equal(baseline, observed['gated_predictions'])
control = pd.read_parquet('/kaggle/working/B556_gated_control.parquet')
np.testing.assert_array_equal(control.id.to_numpy(dtype=str), ids)
np.testing.assert_array_equal(control.model.to_numpy(dtype=str), labels[baseline])
SC = smol['SC']
assert SC.dtype == np.float32 and SC.shape == raw.shape and np.isfinite(SC).all()
np.testing.assert_array_equal(SC[~observed['adaptive_mask']], 0)
# Same FP32 multiplication and promotion order as the actual c66c5 conditional CPU arm.
extended_raw = raw + .03 * SC
q = source_balance(extended_raw)
extended_labels = gated_predict(q, labels, **gate)
claude = gate['CLAUDE_IDX']
nonclaude = np.flatnonzero(~np.char.startswith(labels, 'claude'))
assert len(claude) == 5 and len(nonclaude) == 13
tabular = pfn['proba']
assert tabular.dtype == np.float64 and tabular.shape == q.shape and np.isfinite(tabular).all()
candidate, prediction, reverted = protected(q[None], tabular, extended_labels[None], claude, nonclaude, weight=.10)
submission = pd.DataFrame(dict(id=ids, model=labels[prediction[0]]))
assert submission.id.is_unique and len(submission) == len(control) and submission.model.isin(labels).all()
submission.to_parquet('/kaggle/working/submission.parquet', index=False)
Path('/kaggle/working/protected_residual_audit.json').write_text(json.dumps(dict(
    status='SMOL_CENTER_ADD_PROTECTED_PFN10_PREDICTIONS_COMPLETE', baseline_balance_and_gate_replay_bitexact=True,
    formula='original_raw + .03*SC -> original_alpha.5_balance -> original_Claude_gate -> protected_PFN.10',
    source_notebook_id=139200100, source_version=2, coefficient=.03, alpha=.5, post=0,
    residual_weight=.10, query_rows=len(ids), extra_global_normalization=False,
    protected_family_membership_relative_to='SmolCENTER extended original gate decisions',
    reverted_family_flips=int(reverted.sum()), training=False), indent=2))
