'''Paired no-model tests for coverage acceptance, slot preservation and bounded 3-cut enumeration.'''
import ast
import json
import importlib.util
import os
import tempfile
import time
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from build_evidence_gap_iterations import COMMON,COVERAGE,THREE,build
from build_evidence_routing_research import FRAMEWORK,REPORTING,SOURCE,router_method,build as base_build
from build_aggressive_fusion_experiments import METRIC_HELPER
from build_flow_experiments import ROOT

original = json.loads(SOURCE.read_text())
env = {'__name__':'__main__'}
exec(FRAMEWORK+router_method(''.join(original['cells'][33]['source']))+REPORTING+METRIC_HELPER+COMMON+COVERAGE+THREE,env)
spec = importlib.util.spec_from_file_location('real_gl',ROOT/'archive/simulator-interface-2026-10-09/gl_fuse.py')
gl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gl)
spec = importlib.util.spec_from_file_location('real_ice',ROOT/'archive/simulator-interface-2026-10-09/ice_fuse.py')
ice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ice)
frame = pd.DataFrame([dict(molecule_id='m', ms2_mzs=[10.,20.],ms2_normalized_intensities=[1.,.5],
    precursor_mz=100.,adduct='[M+H]+',ionization_mode='positive',instrument_type='timsTOF',collision_energy_ev=[20.])])
scope = dict(te=frame,IS_RERUN=False,PREFUSION_LISTS={'m':['A','B','C','D','E']},
    ENG={'m':dict(smiles=['C','B','A','D','E'])},metric_fusion=env['metric_fusion'],
    metric_identity=lambda s:s,chem=SimpleNamespace(score_key=lambda s:s),
    ALPHA=.6,KRR=3.,ICE_LAM=1.,GL_LAM=1.,GL_PKG='/gl',ICE_PKG='/ice',
    ICE_SCORES={'m':{'A':0.,'B':None,'C':None}},gl_of=lambda mid:dict(A=0.,D=1.,E=0.),
    _form2=lambda s:'F' if s in ['A','B','C'] else 'G',gl_fuse=gl,ice_fuse=ice,FRAG_GAP={},
    safe_extend=lambda first,extra:list(dict.fromkeys(first+extra))[:25])
iteration = env['FusionFirstCoverage'](SimpleNamespace(scope=scope))
assert iteration.complete_group([0,1],['A','B'],['a','b'],{'A':.1},{'A':.9}) == {}
assert iteration.complete_group([0,1],['A','B'],['a','b'],{'A':.1},{'A':.9,'B':.3}) == {'A':.1,'B':.3}

def write_control(scope, iteration):
    c,s,k,f = iteration.fused('m')
    o = iteration.baseline_order('m',c,s,k,f)
    control = pd.DataFrame([('m',';'.join(c[i] for i in o)),('placeholder','CCO')], columns=['molecule_id','smiles'])
    control.to_csv('submission.csv',index=False)
    return control

cwd = os.getcwd()
with tempfile.TemporaryDirectory() as directory:
    try:
        os.chdir(directory)
        control = write_control(scope,iteration)
        original_bytes = Path('submission.csv').read_bytes()
        iteration.prepare(control)
        with patch.object(gl,'run_gl',return_value={'m':dict(B=0.,C=10.)}) as run_gl:
            result = iteration.predict()
            assert result['gl']['m']['C'] == 10.
            args,kwargs = run_gl.call_args
            assert args[1][0]['cands'] == ['A','B','C']
            assert all(s['adduct']=='[M+H]+' for s in args[1][0]['spectra'])
            assert kwargs['budget_s'] == 120
        iteration.predict = lambda:dict(gl={'m':dict(A=.9,B=0.,C=10.)},ice={})
        report = iteration.run()
        assert report['details']['accepted_groups'] == 1 and report['changed_rows'] == 1
        assert Path('submission_0441_control.csv').read_bytes() == original_bytes
        result = pd.read_csv('submission.csv')
        before = control.iloc[0].smiles.split(';')
        after = result.iloc[0].smiles.split(';')
        assert [i for i,s in enumerate(before) if s in ('D','E')] == [i for i,s in enumerate(after) if s in ('D','E')]
        assert result.iloc[1].smiles == 'CCO'
        # Incomplete predictions cause exact molecule rollback, not zero-filled scoring.
        control.to_csv('submission.csv',index=False)
        incomplete = env['FusionFirstCoverage'](SimpleNamespace(scope=scope))
        incomplete.predict = lambda:dict(gl={'m':dict(B=1.)},ice={})
        assert incomplete.run()['changed_rows'] == 0

        # Three-cut scoring requests the promised coefficients/cost cap, rejects large structures.
        class FragmentModule:
            def mol_graph(self, smile):
                return np.ones(4),np.arange(6),np.arange(6),1
            def score_molecule(self,spectra,candidates,cfg,cache):
                assert cfg == dict(max_breaks=3,max_bonds_3=24,w2=.6,w3=.3)
                return [float(s=='C')*10 for s in candidates]
        scope.update(gl_of=lambda mid:{},ICE_SCORES={'m':dict(A=None)},
            FRAG_GAP={'m':dict(A=1.,B=0.,C=0.,D=1.,E=0.)},frag_rescore=FragmentModule(),_fcache={})
        three = env['ThreeBondFragmentFallback'](SimpleNamespace(scope=scope))
        three.cache,three.errors = {},[]
        assert three.score([],['A','C'],time.monotonic()-1) == {}
        control = write_control(scope,three)
        report = three.run()
        assert report['details']['accepted_groups'] == 2
        assert pd.read_csv('submission.csv').iloc[1].smiles == 'CCO'
        # Large graph: leave the complete group at its baseline 2-cut order.
        scope['frag_rescore'].mol_graph = lambda s:(np.ones(2),np.arange(25),np.arange(25),1)
        control.to_csv('submission.csv',index=False)
        assert env['ThreeBondFragmentFallback'](SimpleNamespace(scope=scope)).run()['changed_rows'] == 0
        # Any non-null ICE evidence excludes the whole molecule, matching v7c eligibility.
        scope['ICE_SCORES'] = {'m':{'A':1.}}
        control = write_control(scope,three)
        assert env['ThreeBondFragmentFallback'](SimpleNamespace(scope=scope)).run()['details']['target_groups'] == 0
    finally:
        os.chdir(cwd)

# Test real upstream enumeration: a tetrahedral K4 graph needs 3 cuts to isolate an atom.
fragment_source = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(''.join(original['cells'][32]['source'])))
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FRAG_SRC' for t in n.targets))
functions = [n for n in ast.parse(fragment_source).body if isinstance(n,ast.FunctionDef) and n.name in ('_find','_emit_components','_enumerate')]
kernel = {'np':np}
exec(ast.unparse(ast.Module(body=functions,type_ignores=[])),kernel)
ea,eb = np.array([0,0,0,1,1,2]),np.array([1,2,3,2,3,3])
weights = np.array([12.,13.,14.,15.])
two,_ = kernel['_enumerate'](4,weights,ea,eb,2,1)
three,levels = kernel['_enumerate'](4,weights,ea,eb,3,1)
assert len(two) == 0 and 12. in three and np.all(levels == 3)

base = base_build()[0]
for nb in build():
    assert nb['cells'][1:len(base['cells'])] == base['cells'][1:]
    assert nb['metadata']['kaggle'] == base['metadata']['kaggle']
    assert all(c['cell_type']=='code' and c['execution_count'] is None and not c['outputs'] for c in nb['cells'])
print('PASS: Initial01 prefix, paired coverage rollback, finite-score retention, protected slots, 3-cut kernel, cost cap and molecule eligibility')
