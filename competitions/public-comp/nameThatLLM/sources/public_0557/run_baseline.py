"""Exact frozen B556 with one reversible observation hook and post-execution exports."""
import hashlib
import json
import os
from pathlib import Path
from capture_runtime import wrap_fit_head, capture_balanced

assert os.name != 'nt' and Path('/kaggle/input').is_dir()
here = Path(__file__).resolve().parent
config = json.loads((here / 'runtime_config.json').read_text())
raw = (here / 'ntl_reference.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == config['reference_source_sha256']
source = raw.decode('utf-8')
assert config['baseline_balance_alpha'] == .5
assert config['reference_notebook_id'] == 139200100 and config['reference_version'] == 2
assert 'BALANCE_ALPHA = 0.5' in source and "W = {'stylegram': 1.516" in source
loop = 'for key, mt, hid, layers, C, smoke_id, pool in LLM_HEADS:'
hook = 'fit_head = _tabular_wrap_fit_head(fit_head, train, test, LABELS)\n' + loop
assert source.count(loop) == 1
modified = source.replace(loop, hook)
assert modified.replace(hook, loop) == source
namespace = dict(__name__='__main__', __file__=str(here / 'ntl_reference.py'),
                 _tabular_wrap_fit_head=wrap_fit_head)
exec(compile(modified, 'B556_observation_only', 'exec'), namespace)
assert Path('/kaggle/working/submission.parquet').is_file()
import numpy as np
cache = Path(os.environ['NTL_TABULAR_TEMP']).resolve()
assert cache.is_relative_to(Path('/kaggle/temp').resolve())
P, raw_logits, test, labels = (namespace[key] for key in ('P', 'LOGIT', 'test', 'LABELS'))
groups, adaptive, pred = (namespace[key] for key in ('G_TE', 'ADAPT_ROWS', 'pred'))
assert P.dtype == raw_logits.dtype == np.float64 and P.shape == raw_logits.shape == (len(test), 18)
assert groups.shape == adaptive.shape == pred.shape == (len(test),)
assert np.issubdtype(groups.dtype, np.integer) and adaptive.dtype == bool
sizes = np.bincount(groups)[groups]
np.testing.assert_array_equal(adaptive, (sizes >= 2) & (sizes <= 18))
# Original code has already finished. Extra I/O observes values without rerunning its gate or models.
capture_balanced(P, test, labels, namespace['DC_TE'], namespace['LC_TE'], namespace['CLAUDE_IDX'],
                 namespace['CLAUDE_GATE_LAM'], namespace['CLAUDE_LORA_LAM'])
np.savez_compressed(cache / 'baseline_observation.npz', ids=test.id.to_numpy(dtype=str),
    labels=np.asarray(labels, dtype=str), raw=raw_logits, balanced=P, gated_predictions=pred,
    recovered_group_ids=groups, adaptive_mask=adaptive, pool_ids=np.zeros(len(test), dtype=np.int64))
Path('/kaggle/working/B556_source_parity.json').write_text(json.dumps(dict(
    source_notebook_id=139200100, source_version=2, source_script_version_id=359313348,
    original_source_sha256=hashlib.sha256(raw).hexdigest(), executed_source_sha256=hashlib.sha256(modified.encode()).hexdigest(),
    observation_hook_reversible=True, original_W_TWIN_and_all_scientific_bodies_unchanged=True,
    original_balance_alpha=.5, original_stylegram_lambda=1, original_gate_last=True,
    exported_values=['unbalanced_raw', 'balanced_P', 'original_gated_labels', 'actual_recovered_groups', 'actual_adaptive_mask'],
    query_pool_semantics='One actual complete test CSV pool; pool_id0; source G_TE is reused without reclustering',
    observation_arrays_saved_under_temp_only=True, extra_language_model_passes=0), indent=2))
