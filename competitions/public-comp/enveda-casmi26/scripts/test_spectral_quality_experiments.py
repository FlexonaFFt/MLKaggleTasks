'''Scoring parity, bounded aggregation, bilateral precursor masks and complete-score fallback.'''
import ast
import json
from pathlib import Path
import numpy as np
from build_spectral_quality_experiments import SCORING, RERUN, build
from build_ice_fallback_versions import build as base_build
from build_flow_experiments import ROOT

env = {}
exec(SCORING, env)
weights = env['quality_weights']([
    dict(mz=[10,20,30], it=[1,1,1], prec=100, base_peak_intensity=100, ce_ev=[20]),
    dict(mz=[10], it=[1], prec=100, base_peak_intensity=1, ce_ev=[20,40,60,80])])
assert weights[0] > weights[1] and np.all((weights >= .5) & (weights <= 2.))
assert env['quality_weights']([]).size == 0
assert env['quality_weights']([dict(mz=[], it=[], prec=100)]).tolist() == [1.]
mz, it = env['precursor_mask']([10,99.99,100,100.01,101], [1]*5, 100)
assert mz.tolist() == [10.,101.] and len(mz) == len(it)

def prep(mz, it, *args):
    mz, it = np.asarray(mz, float), np.asarray(it, float)
    return mz, it / it.sum() if it.sum() else it
def prep_spectrum(sp):
    return prep(sp['mz'], sp['it'])
def original_plan(mol):
    return [(*prep_spectrum(sp), 'H', 'T', sp.get('energy', 20.)) for sp in mol['spectra']]
def similarity(qm, qp, pm, pp, *args):
    return float(qp.sum() + pp.sum() + (100 in qm) + (100 in pm))
def make_scope():
    return dict(molecule_plan=original_plan, is_covered=lambda sp:True,
        prep_query=prep, prep_spectrum=prep_spectrum, entropy_sim=similarity,
        SUM_CE=[10,20], merge=lambda parts:parts[0], FLOOR=.002, TOPK=256, POWER=1., ENTW=True, TOL=.01, PPM=20.)
spectra = [dict(mz=[10,20,30,40,50,100], it=[1]*6, prec=100),
           dict(mz=[10,20,30,40,50,60], it=[1]*6, prec=100)]
mol = {'spectra':spectra}
cache = {('A','H','T',20.): ([10,20,30,40,50,100], [1]*6)}
scope = make_scope()
env['install'](scope, 'precursor_masked')
plan = scope['molecule_plan'](mol)
assert scope['score_molecule'](plan, ['A','missing'], cache) == {'A':2.,'missing':None}
# A sparse prediction must retain both original sides, not mask just the observed one.
sparse = {('A','H','T',20.): ([10,100], [1,1])}
assert scope['score_molecule'](plan, ['A'], sparse)['A'] == 3.5
scope = make_scope()
env['install'](scope, 'quality_weighted')
plan = scope['molecule_plan'](mol)
assert scope['score_molecule'](plan, ['A'], cache)['A'] == 3.5  # identical quality -> original mean
mol['spectra'][0]['base_peak_intensity'] = 100
mol['spectra'][1]['base_peak_intensity'] = 1
plan = scope['molecule_plan'](mol)
assert scope['score_molecule'](plan, ['A'], cache)['A'] > 3.5
# Unknown energy uses the inherited summed prediction and missing CE parts stay unavailable.
unknown = {'spectra':[dict(spectra[0], energy=float('nan'))]}
plan = scope['molecule_plan'](unknown)
ce_cache = {('A','H','T',10):cache[('A','H','T',20.)], ('A','H','T',20):cache[('A','H','T',20.)]}
assert scope['score_molecule'](plan, ['A'], ce_cache)['A'] == 4.
assert scope['score_molecule'](plan, ['A'], {})['A'] is None

# Check installed-runner assumptions and parity against BOTH downloaded upstream scoring functions.
for runner in ('gl_runner.py', 'ice_runner.py'):
    code = (ROOT / 'archive/simulator-interface-2026-10-12' / runner).read_text()
    functions = {n.name:n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    assert 'np.mean(sims)' in ast.unparse(functions['score_molecule'])
    assert 'for _, _, ad, ins, ceb in plan' in ast.unparse(functions['molecule_jobs'])
    upstream = dict(make_scope(), np=np)
    exec(ast.unparse(functions['score_molecule']), upstream)
    equal_mol = {'spectra':[dict(mz=[10,20,30,40,50,100], it=[1]*6, prec=100)]*2}
    expected = upstream['score_molecule'](original_plan(equal_mol), ['A'], cache)
    env['install'](upstream, 'quality_weighted')
    assert upstream['score_molecule'](upstream['molecule_plan'](equal_mol), ['A'], cache) == expected

# Validate complete-molecule rollback independently of the model runner.
nodes = [node for node in ast.parse(RERUN).body if isinstance(node, ast.FunctionDef) and node.name == 'keep_complete']
exec(ast.unparse(ast.Module(body=nodes, type_ignores=[])), env)
original = {'m':{'A':1.,'B':2.,'C':None}}
assert env['keep_complete'](original, {'m':{'A':3.}}) == (original, 0)
new, n = env['keep_complete'](original, {'m':{'A':3.,'B':4.,'C':5.}})
assert n == 1 and new['m'] == {'A':3.,'B':4.,'C':None} and original['m']['A'] == 1.

base = base_build()[0]
for nb in build():
    assert nb['cells'][1:len(base['cells'])] == base['cells'][1:]
    assert nb['metadata']['kaggle'] == base['metadata']['kaggle']
    ast.parse(SCORING)
print('PASS: E006 prefix parity, bounded weights, masked normalization, sparse fallback, unknown CE and coverage rollback')
