"""Optional owned final Smol feature pass and exact pool-local CENTER readout."""
import hashlib
import json
import os
from pathlib import Path
import time

here = Path(__file__).resolve().parent
from admission_preflight import validate
config = validate(here)
assert config['candidate_admitted'] and config['selected_by_root']
assert config['assets_public_audited'] and config['diagnostic_actual_PASS_bound']
assert config['raw_extension'] == 'smol_center_add' and config['coefficient'] == .03
assert os.name != 'nt' and Path('/kaggle/input').is_dir()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


for name, digest in config['source_sha256'].items():
    assert sha(here / name) == digest, name
cache = Path(os.environ['NTL_TABULAR_TEMP']).resolve()
assert cache.is_relative_to(Path('/kaggle/temp').resolve())
assert not (cache / 'smol_center_add.npz').exists(), 'No automatic reuse/retry'
spec = json.loads((here / 'smol-spec.json').read_text())
assert spec['inference_authorized'] and not spec['diagnostic_only_authorized']
assert spec['actual_final_binding']['paired_gain_admitted_by_root']
assets = [p for p in Path('/kaggle/input').rglob('asset_manifest.json')
          if sha(p) == config['compact_asset_manifest_sha256']]
stages = [p for p in Path('/kaggle/input').rglob('smollm3_staging_complete.json')
          if sha(p) == spec['base_manifest_sha256']]
trains = [p for p in Path('/kaggle/input').rglob('train.csv') if sha(p) == spec['train_sha256']]
tests = list(Path('/kaggle/input').rglob('test.csv'))
assert len(assets) == len(stages) == len(trains) == len(tests) == 1

import numpy as np
import pandas as pd
from component import SmolFinalPlain
from center_readout import lora_readouts

with np.load(cache / 'baseline_observation.npz', allow_pickle=False) as values:
    observed = {key: values[key].copy() for key in values.files}
query = pd.read_csv(tests[0], usecols=['id', 'text'], dtype=str)
np.testing.assert_array_equal(query.id.to_numpy(dtype=str), observed['ids'])
assert query.id.is_unique and len(query) > 0
model = SmolFinalPlain(here / 'smol-spec.json', assets[0].parent, stages[0].parent, trains[0])
np.testing.assert_array_equal(model.readouts['classes'], observed['labels'])
import torch
assert torch.cuda.device_count() == 1 and os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
query_started = time.monotonic()
begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
begin.record()
# One query forward pass; _features restores the actual CSV order after stable length sorting.
Z = model._features(query.text.tolist())
end.record()
torch.cuda.synchronize()
query_cuda_ms = float(begin.elapsed_time(end))
assert query_cuda_ms > 0
assert Z.dtype == np.float32 and Z.shape == (len(query), 12288)
pools = observed['pool_ids'][None]
groups = observed['recovered_group_ids'][None]
adaptive = observed['adaptive_mask'][None]
assert pools.shape == groups.shape == adaptive.shape == (1, len(query))
np.testing.assert_array_equal(pools, np.zeros_like(pools))
# The c66c5 body centers separately within each actual recovered adaptive group/pool.
# Its unused plain return is never added to the ensemble and triggers no extra model pass.
_, centered, _ = lora_readouts(Z, model.readouts, pools, groups, adaptive)
SC = centered[0]
assert SC.dtype == np.float32 and SC.shape == (len(query), 18) and np.isfinite(SC).all()
np.testing.assert_array_equal(SC[~adaptive[0]], 0)
np.savez_compressed(cache / 'smol_center_add.npz', ids=observed['ids'], labels=observed['labels'],
                    SC=SC, pool_ids=pools[0], recovered_group_ids=groups[0], adaptive_mask=adaptive[0])
Path('/kaggle/working/smol_center_inference_audit.json').write_text(json.dumps(dict(
    status='OWNED_FINAL_CENTER_FEATURE_PASS_COMPLETE', query_feature_passes=1, query_rows=len(query),
    raw_feature_shape=list(Z.shape), raw_feature_dtype=str(Z.dtype), anchor_replay=model.anchor_replay,
    CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], PID=os.getpid(),
    logical_gpu_name=torch.cuda.get_device_name(0), query_cuda_event_ms=query_cuda_ms,
    query_wall_seconds=time.monotonic()-query_started,
    center_helper_sha256=sha(here / 'center_readout.py'), groups_reclustered=False,
    centers_within_actual_pools_and_recovered_groups=True, zero_outside_adaptive=True,
    uses_plain_readout=False, training=False, target_column_read=False), indent=2))
