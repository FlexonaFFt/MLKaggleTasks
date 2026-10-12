'''Shared-scale features, metric deduplication and independent paired-overlay regression checks.'''
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from build_pubchem_retrieval_experiments import HELPERS, WORKER, RUN, build
from build_ice_fallback_versions import build as base_build
from build_aggressive_fusion_experiments import METRIC_HELPER

env = {}
exec(HELPERS + METRIC_HELPER, env)
identity = lambda s: None if s == 'invalid' else s.split('-')[0]
union = env['polarity_union']
assert union(['A','A-alt'], ['B','B-alt'], ['B'], identity) == ['A','B']
assert union([], ['B'], ['B-alt'], identity) == ['B']
assert union(['A'], [], [], identity) == ['A']
common = env['joint_common_score']
assert np.argmax(common([1,10,1], [0,0,0], [0,0,0])) == 1
assert np.isfinite(common([1,1], [0,0], [0,0])).all()
try:
    common([1,np.nan], [0,0], [0,0])
except ValueError:
    pass
else:
    raise AssertionError('Reject non-finite candidate evidence')
tree = ast.parse(WORKER)
worker_env = {}
exec(ast.unparse(ast.Module(body=tree.body[:-1], type_ignores=[])), worker_env)
frame = pd.DataFrame([dict(ms2_mzs=[1.,2.], ms2_normalized_intensities=[1.,.5],
    ionization_mode='negative', adduct='[M-H]-', precursor_mz=100., collision_energy_ev=[20,40])])
assert worker_env['spectra_for'](frame)[0]['mode'] == -1
assert worker_env['spectra_for'](frame)[0]['ce_n'] == 2

baseline = base_build()[0]
for nb in build():
    assert nb['cells'][1:len(baseline['cells'])] == baseline['cells'][1:]
    assert nb['metadata']['kaggle'] == baseline['metadata']['kaggle']
    ast.parse(''.join(nb['cells'][-1]['source']))

def extend(first, extra, limit=25):
    out, seen = [], set()
    for s in first+extra:
        if identity(s) not in seen:
            seen.add(identity(s)); out.append(s)
        if len(out) == limit:
            break
    return out
def merge(first, keys, pc, pc_keys, slots):
    return extend(first[:1], pc + first[1:])
def promote(first, keys, pc, pc_keys, slots):
    return extend(pc[:1], first + pc[1:])
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as folder:
    try:
        os.chdir(folder)
        control = pd.DataFrame([('m','A;B'), ('strong','A;B'), ('single','A;B')], columns=['molecule_id','smiles'])
        for mode in ('dual', 'joint'):
            control.to_csv('submission_e006_control.csv', index=False)
            scope = dict(env, pd=pd, np=np, json=json, Path=Path, MODE=mode, _control=control,
                _base={'m':(['A','B'],['A','B'],.1)}, _pc={'m':dict(pc=['P'], fz_top=0.,pc_fz=[0.],best_pool_fz=0.)},
                PREFUSION_LISTS={'m':['A','P','B']}, ENG={'m':dict(smiles=['C'])}, ALPHA=.6,KRR=3.,
                SLOTS_AGG=[2],SLOTS_GENTLE=[4],REL_TH=600.,ICE_LAM=1.,GL_LAM=1.,
                metric_identity=identity, chem=SimpleNamespace(score_key=identity), _form2=lambda s:'F',
                safe_extend=extend, merge=merge,promote=promote, ICE_SCORES={},gl_of=lambda mid:{},
                sha256=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
            if mode == 'dual':
                scope.update(_spec={'m':{}}, _retrieved={'m':dict(positive=dict(smiles=['X']),negative=dict(smiles=['X']))})
            else:
                scope.update(_spec={'m':dict(smiles=['A','B','X'])}, _support={'m':[0.,0.,0.]},
                    _retrieved={'m':dict(smiles=['A','B','X'],fingerprint=[0.,0.,100.],popularity=[0.,0.,0.])})
            exec(RUN, scope)
            output = pd.read_csv('submission.csv')
            assert output.iloc[1:].equals(control.iloc[1:])
            assert 'X' in output.iloc[0].smiles.split(';')
            scope['_retrieved'] = {}
            exec(RUN, scope)
            assert pd.read_csv('submission.csv').equals(control)
        # Promotion remains the same candidate even when the split retrieval prefers another.
        scope.update(MODE='dual', _spec={'m':{}},
            _retrieved={'m':dict(positive=dict(smiles=['X']),negative=dict(smiles=['X']))},
            PREFUSION_LISTS={'m':['P','A','B']})
        exec(RUN, scope)
        assert pd.read_csv('submission.csv').iloc[0].smiles.split(';')[0] == 'P'
    finally:
        os.chdir(cwd)
print('PASS: preserved E006 prefix/inputs, worker syntax, polarity spectra, common-score scale, dedup, paired overlays, promotion anchor, no-op fallback')
