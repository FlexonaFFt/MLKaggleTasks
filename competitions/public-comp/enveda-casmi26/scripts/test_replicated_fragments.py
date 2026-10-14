'''Execute the exact new scorer with synthetic mass matches; no GPU or RDKit substitution.'''
import math
import time
from types import SimpleNamespace
import numpy as np
from build_discriminative_evidence import COMMON
from build_replicated_fragments import REPLICATED,build
from build_evidence_routing_research import build as initial_build

env = dict(np=np,math=math,time=time)
exec(COMMON+REPLICATED,env)
module = SimpleNamespace(
    mol_graph=lambda s:(np.ones(2),np.zeros(1),np.zeros(1),1),
    resolve_cfg=lambda cfg:dict(w1=1.,w2=.6,tol=.005),
    candidate_levels=lambda s,cfg,cache:[np.array([10. if s=='A' else 20.]),np.array([]),np.array([])],
    _prep_spectrum=lambda s,cfg:(np.array(s['intensity']),np.array([10.,10.,20.,20.]),1,4))
scorer = env['ReplicatedFragmentFallback'](SimpleNamespace(scope={'frag_rescore':module}))
scorer.cache,scorer.errors = {},[]
spectra = [dict(intensity=[.49,.49,.01,.01]),dict(intensity=[.05,.05,.45,.45]),dict(intensity=[.06,.06,.44,.44])]
assert scorer.score(spectra[:2],['A','B'],time.monotonic()+10) == {}
assert scorer.score([spectra[0]]*3,['A','B'],time.monotonic()+10) == {}
scores = scorer.score(spectra,['A','B'],time.monotonic()+10)
assert np.isclose(scores['A'],.12) and np.isclose(scores['B'],.88)
assert scorer.score(spectra+spectra,['A','B'],time.monotonic()+10) == scores
assert not scorer.errors
assert scorer.score(spectra,['A','B'],time.monotonic()-1) == {}
nb,base = build(),initial_build()[0]
assert nb['cells'][1:len(base['cells'])] == base['cells'][1:]
for cell in nb['cells']:
    compile(''.join(cell['source']),'notebook-cell','exec')
    assert not cell['outputs'] and cell['execution_count'] is None
print('PASS: real new scorer, median aggregation, duplicate rejection, minimum spectra, deadline, unchanged baseline prefix and cell compilation')
