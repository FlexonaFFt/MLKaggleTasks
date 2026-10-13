'''Verify OOP row parity against uploaded v7c and guard covered-group slots.'''
import ast
import json
import importlib.util
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from build_evidence_routing_research import SOURCE, FRAMEWORK, REPORTING, router_method, build
from build_aggressive_fusion_experiments import METRIC_HELPER
from build_flow_experiments import ROOT
import pandas as pd
from unittest.mock import patch

original = json.loads(SOURCE.read_text())
source = ''.join(original['cells'][33]['source'])
env = {'__name__': '__main__'}
exec(FRAMEWORK + router_method(source) + REPORTING + METRIC_HELPER, env)
# Both supported runtimes reach setup; unrelated runtimes remain rejected.
registry = env['InputRegistry']('runtime test', [(0, 'setup_ran = True')])
for version in ((3,12,0),(3,13,15)):
    runtime_scope = {}
    with patch('sys.version_info', version):
        registry.run(runtime_scope)
    assert runtime_scope['setup_ran'] is True
class Version(tuple):
    major, minor = 3, 14
runtime_scope = {}
with patch('sys.version_info', Version((3,14,0))):
    try:
        registry.run(runtime_scope)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Reject unsupported runtime before setup')
assert 'setup_ran' not in runtime_scope
nodes = [n for n in ast.parse(source).body[0].body if isinstance(n, ast.FunctionDef) and n.name in ('_finite_coverage', 'rerank_row')]
oracle = dict(env)
exec(ast.unparse(ast.Module(body=nodes, type_ignores=[])), oracle)

spec = importlib.util.spec_from_file_location('real_gl_fuse', ROOT / 'archive/simulator-interface-2026-10-09/gl_fuse.py')
gl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gl_module)
spec = importlib.util.spec_from_file_location('real_ice_fuse', ROOT / 'archive/simulator-interface-2026-10-09/ice_fuse.py')
ice_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ice_module)
scope = dict(metric_fusion=env['metric_fusion'], metric_identity=lambda s:s,
    ALPHA=.6,KRR=3.,ICE_LAM=1.,GL_LAM=1.,
    PREFUSION_LISTS={'m':['A','B','C','D']}, ENG={'m':dict(smiles=['A','C','D','B'])},
    chem=SimpleNamespace(score_key=lambda s:s), _form2=lambda s:'F' if s in ('A','B') else 'G',
    gl_fuse=gl_module, ice_fuse=ice_module, FRAG_GAP={}, ICE_SCORES={}, gl_of=lambda mid:{},
    safe_extend=lambda first,extra:list(dict.fromkeys(first+extra))[:25])
Config, Router = env['ExperimentConfig'], env['EvidenceRouter']
class Fragments:
    def __init__(self, evidence):
        self.evidence, self.requested = evidence, []
    def score(self, scope, mid, candidates):
        self.requested.extend(candidates)
        return {s:self.evidence[s] for s in candidates if s in self.evidence}
fragment = Fragments({'A':0.,'B':1.,'C':0.,'D':1.})
router = Router(scope, Config(), fragment)
cases = [({}, {}, {}), ({'A':1.,'B':0.}, {}, {}),
         ({'A':1.,'B':0.}, {'A':0.,'B':1.}, {}),
         ({'A':None,'B':None}, {}, {'m':{'A':0.,'B':1.,'C':1.,'D':0.}})]
for ice, gl, frag in cases:
    scope.update(ICE_SCORES={'m':ice}, gl_of=lambda mid,g=gl:g, FRAG_GAP=frag)
    for active in (False,True):
        expected, audit = oracle['rerank_row'](scope, 'm', ['A','B','C','D'], window=40, ice_fallback=True, processed=active)
        actual, obj_audit = router.rerank_row('m', ['A','B','C','D'], window=40, ice_fallback=True, processed=active)
        assert actual == expected
        assert {k:v for k,v in obj_audit.items() if k != 'group_fragment_groups'} == audit

# A covered F group must never receive fragment evidence; the uncovered G slots can change.
scope.update(ICE_SCORES={'m':{'A':1.,'B':0.}}, gl_of=lambda mid:{'A':0.,'B':1.}, FRAG_GAP={})
cfg = Config(group_fragment_fallback=True, fragment_weight=3.)
router = Router(scope, cfg, fragment)
original_candidates = ['A','B','C','D']
revised, n = router.group_rescore('m', original_candidates, original_candidates, [4.,3.,2.,1.],
    scope['ICE_SCORES']['m'], scope['gl_of']('m'))
assert revised[:2] == ['A','B'] and revised[2:] == ['D','C'] and n == 1
assert set(fragment.requested) == {'C','D'}
# One finite score excludes an entire group; missing fragment member preserves the group.
assert router.group_rescore('m', original_candidates, original_candidates, [4.,3.,2.,1.],
    {'A':1.,'C':1.}, {}) == (original_candidates, 0)
router.fragments = Fragments({'C':1.})
assert router.group_rescore('m', original_candidates, original_candidates, [4.,3.,2.,1.],
    {'A':1.}, {}) == (original_candidates, 0)
# Already-routed whole-molecule fragment lists must not receive a second pass.
scope.update(ICE_SCORES={'m':{'A':None}}, gl_of=lambda mid:{}, FRAG_GAP={'m':dict(A=0.,B=1.,C=0.,D=1.)})
class NoSecondPass:
    def score(self, *args):
        raise AssertionError('Do not repeat the baseline fragment pass')
router = Router(scope, cfg, NoSecondPass())
_, audit = router.rerank_row('m', ['A','B','C','D'], window=40, ice_fallback=True, processed=True, group=True)
assert audit['branch'] == 'frag_gap' and audit['group_fragment_groups'] == 0

for nb in build():
    assert nb['metadata']['kaggle'] == dict(original['metadata']['kaggle'], isGpuEnabled=True)
    assert nb['metadata']['kaggle']['dockerImageVersionId'] == 31430
    assert all(c['cell_type']=='code' and not c['outputs'] and c['execution_count'] is None for c in nb['cells'])
    pack = ast.literal_eval(ast.parse(''.join(nb['cells'][5]['source'])).body[1].value)
    for index, code in pack.items():
        expected = ''.join(original['cells'][index]['source'])
        if index == 8:
            expected = expected.replace("assert platform.python_version_tuple()[:2] == ('3', '12')",
                "assert platform.python_version_tuple()[:2] in (('3', '12'), ('3', '13')), 'Python 3.12/3.13 required'")
            guard = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Assert)
                and 'python_version_tuple' in ast.unparse(n.test))
            for version in (('3','12'),('3','13')):
                assert eval(compile(ast.Expression(guard.test), '<preflight-runtime>', 'eval'),
                    {'platform':SimpleNamespace(python_version_tuple=lambda v=version:v)})
        assert code == expected
        ast.parse(code)
    frag_tree = ast.parse(pack[32])
    embedded = next(ast.literal_eval(n.value) for n in ast.walk(frag_tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='FRAG_SRC' for t in n.targets))
    ast.parse(embedded)

# Execute the entire author-written orchestration on fixtures, including per-row source parity.
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as directory:
    try:
        os.chdir(directory)
        scope.update(ICE_SCORES={'m':{'A':1.,'B':0.}}, gl_of=lambda mid:{'A':0.,'B':1.}, FRAG_GAP={})
        reference_frame = pd.DataFrame([('m','A;B;C;D'),('placeholder','CCO')], columns=['molecule_id','smiles'])
        faithful, _ = oracle['rerank_row'](scope,'m',['A','B','C','D'],window=40,ice_fallback=True,processed=True)
        controls = pd.DataFrame([('m',';'.join(faithful)),('placeholder','CCO')], columns=['molecule_id','smiles'])
        stage = env['SourceStage']('synthetic data', [(0,"te = pd.DataFrame({'molecule_id':['m']})")])
        fragment_stage = env['FragmentEvidenceScorer']('synthetic fragment module', [])
        fragment_stage.score = lambda scope,mid,candidates:{s:float(s=='D') for s in candidates}
        reference_stage = env['SourceStage']('synthetic source output', [])
        runner = env['CasmiExperiment'](Config(), [stage], fragment_stage, reference_stage)
        runner.scope.update(scope, pd=pd)
        reference_frame.to_csv('submission_reference.csv',index=False)
        controls.to_csv('submission.csv',index=False)
        # Charts tested separately so this check requires no local plotting dependency.
        plot = env['ExperimentAudit'].plot
        env['ExperimentAudit'].plot = lambda self, active:None
        result = runner.run()
        assert result['changed_rows'] == 0 and pd.read_csv('submission.csv').equals(controls)
        runner.config = Config(group_fragment_fallback=True, fragment_weight=3.)
        controls.to_csv('submission.csv',index=False)
        result = runner.run()
        assert result['changed_rows'] == 1 and result['group_fallback_count'] == 1
        assert pd.read_csv('submission.csv').iloc[1].smiles == 'CCO'
        env['ExperimentAudit'].plot = plot
    finally:
        os.chdir(cwd)
print('PASS: all source stages preserved, OOP/original row parity, group-slot protection, partial-score exclusion and no double fragment pass')
