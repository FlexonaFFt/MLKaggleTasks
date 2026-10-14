'''No-GPU checks for the actual OOP launch classes and credited peak-matching kernel.'''
import ast
import json
import math
import importlib.util
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
from build_discriminative_evidence import COMMON, COVERAGE, CONSENSUS, DISCRIMINATIVE, build
from build_evidence_routing_research import SOURCE, build as initial_build
from build_flow_experiments import ROOT

env = dict(np=np,pd=pd,math=math,time=time)
exec(COMMON+COVERAGE+CONSENSUS+DISCRIMINATIVE,env)
frame = pd.DataFrame([dict(molecule_id='m',ionization_mode='positive',adduct='[M+H]+')])
scope = dict(te=frame,IS_RERUN=False,ICE_SCORES={'m':dict(A=.1,B=.8)},
    gl_of=lambda mid:dict(A=.1,B=.9),safe_extend=lambda first,tail:list(dict.fromkeys(first+tail))[:25])
control = pd.DataFrame([('m','A;X;B'),('inactive','CCO')],columns=['molecule_id','smiles'])
check = env['SimulatorConflictCheck'](SimpleNamespace(scope=scope))
check.fused = lambda mid:(['A','X','B'],[1.,.8,.7],['A','X','B'],['F','G','F'])
check.baseline_order = lambda *args:[0,1,2]
check.control = lambda:control
check.predict = lambda:{'ice':{}}
check.finish = lambda rows,control,name,details:(rows,details)
rows,details = check.run()
assert rows == [('m','B;X;A'),('inactive','CCO')]
assert details['accepted_groups'] == 1
# Ties, disagreement and missing scores preserve every baseline slot.
for values in (dict(A=.8,B=.8),dict(A=.9,B=.1),dict(A=.1,B=None)):
    scope['ICE_SCORES']['m'] = values
    rows,details = check.run()
    assert rows == [('m','A;X;B'),('inactive','CCO')]
    assert details['accepted_groups'] == 0
# Fresh complete-pass evidence is used instead of mixing old/new scales.
scope['ICE_SCORES']['m'] = dict(A=99.,B=None)
check.predict = lambda:{'ice':{'m':dict(A=.1,B=.9)}}
assert check.run()[0][0] == ('m','B;X;A')
scope['gl_of'] = lambda mid:dict(A=.9,B=.9)
assert check.run()[1]['target_groups'] == 0

fragment = env['DiscriminativeFragmentFallback'](SimpleNamespace(scope={}))
weights = np.array([.4,.3,.3])
credits = np.array([[1.,1.,0.],[1.,0.,.6]])
assert np.allclose(fragment.contrast(credits,weights),[.3,.18])
assert fragment.contrast(np.ones((2,3)),weights) is None
assert fragment.contrast(np.array([[1.,1.,0.],[1.,0.,0.]]),weights) is None
assert fragment.contrast(credits,np.array([.98,.01,.01])) is None

# Execute real upstream cfg/adduct/neutral-mass preparation functions; fake only molecular graphs.
nb = json.loads(SOURCE.read_text())
tree = ast.parse(''.join(nb['cells'][32]['source']))
source = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='FRAG_SRC' for t in n.targets))
nodes = [n for n in ast.parse(source).body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign))
    or isinstance(n,ast.FunctionDef) and n.name in ('resolve_cfg','_carriers','_prep_spectrum')]
kernel = {}
exec(ast.unparse(ast.Module(body=nodes,type_ignores=[])),kernel)
module = SimpleNamespace(**kernel)
module.mol_graph = lambda smile:(np.ones(3),np.zeros(2),np.zeros(2),1)
module.candidate_levels = lambda smile,cfg,cache:[np.array([100.]),np.array([20.,30.] if smile=='A' else [40.,50.]),np.array([])]
fragment.scope = {'frag_rescore':module}
fragment.cache,fragment.errors = {},[]
spectrum = dict(mz=np.array([100.,20.,30.,40.,50.])+kernel['PROTON'],
    it=np.array([1.,.5,.5,.1,.1]),adduct='[M+H]+',mode=1,prec=100.+kernel['PROTON'])
values = fragment.score([spectrum],['A','B'],time.monotonic()+10)
assert values['A'] > values['B'] > 0 and not fragment.errors
assert fragment.score([spectrum],['A','B'],time.monotonic()-1) == {}
module.mol_graph = lambda smile:(np.ones(3),np.zeros(41),np.zeros(41),1)
assert fragment.score([spectrum],['A','B'],time.monotonic()+10) == {}

# Exercise final fragment routing with the real same-formula reranker, not a mock sort.
spec = importlib.util.spec_from_file_location('gl_fuse',ROOT/'archive/simulator-interface-2026-10-09/gl_fuse.py')
gl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gl)
frame['ms2_mzs'] = [[10.,20.]]
frame['ms2_normalized_intensities'] = [[.5,.5]]
frame['precursor_mz'] = 100.
scope.update(ICE_SCORES={'m':{}},gl_of=lambda mid:{},gl_fuse=gl)
fragment.scope = scope
fragment.fused = lambda mid:(['A','X','B','C'],[1.,.8,.99,0.],['A','X','B','C'],['F','G','F','F'])
fragment.baseline_order = check.baseline_order
fragment.baseline_order = lambda *args:[0,1,2,3]
fragment_control = pd.DataFrame([('m','A;X;B;C'),('inactive','CCO')],columns=['molecule_id','smiles'])
fragment.control = lambda:fragment_control
fragment.finish = check.finish
fragment.score = lambda spectra,candidates,deadline:dict(A=0.,B=1.,C=0.)
rows,details = fragment.run()
assert rows == [('m','B;X;A;C'),('inactive','CCO')] and details['accepted_groups'] == 1
# The new weight must allow a genuine pair reversal, unlike the original lambda .5.
fragment.fused = check.fused
fragment.baseline_order = check.baseline_order
fragment.control = lambda:control
fragment.score = lambda *args:dict(A=0.,B=1.)
assert fragment.run()[0] == [('m','B;X;A'),('inactive','CCO')]
fragment.score = lambda *args:{}
assert fragment.run()[0] == [('m','A;X;B'),('inactive','CCO')]
scope['ICE_SCORES']['m'] = dict(A=.1)
assert fragment.run()[1]['target_groups'] == 0

base = initial_build()[0]
for notebook in build():
    assert notebook['cells'][1:len(base['cells'])] == base['cells'][1:]
    assert notebook['metadata']['kaggle'] == base['metadata']['kaggle']
    for cell in notebook['cells']:
        assert not cell['outputs'] and cell['execution_count'] is None
        compile(''.join(cell['source']),'notebook-cell','exec')
print('PASS: agreement promotion, ties/missing/disagreement rollback, protected slots, complete-pass scaling, real adduct matching, contrast gates, cost cap, unchanged control prefix')
