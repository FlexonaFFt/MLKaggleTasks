"""Run with python3 test_candidate_experiments.py; no Kaggle data needed."""
import ast
import json
import time
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from build_candidate_experiments import ROOT, SOURCE, build_aligned, build_expansion, text


def definitions(source, names, namespace):
    tree = ast.parse(source)
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), '<notebook definitions>', 'exec'), namespace)


def check_gate():
    nb = build_aligned()
    ns = dict(np=np, LIB_TAU=0.9, chem=SimpleNamespace(score_key=lambda s: s))
    definitions(text(nb, 25), {'align_candidate_evidence', 'PubChemDualConfidenceGate'}, ns)
    gate = ns['PubChemDualConfidenceGate']()
    evidence = dict(pc=['weak', 'strong'], pc_keys=['weak', 'strong'],
        best_pool_fz=100.0, fz_top=1000.0, S=8.0, top_pop=10.0,
        candidate_evidence=dict(weak=dict(raw_fz=110.0, score=1.0, popularity=0.0),
                                strong=dict(raw_fz=1000.0, score=8.0, popularity=10.0)))
    base, keys, slots = ['base'], ['base'], [4, 8, 12, 16, 20]
    old, reason = gate.apply(base, keys, 0.2, evidence, slots)
    assert old[0] == 'weak' and reason == 'both', 'reproduce stale-evidence promotion'
    fixed = ns['align_candidate_evidence'](evidence)
    new, reason = gate.apply(base, keys, 0.2, fixed, slots)
    assert new[0] == 'base' and reason is None
    reordered = dict(evidence, pc=['strong', 'weak'], pc_keys=['strong', 'weak'])
    new, reason = gate.apply(base, keys, 0.2, ns['align_candidate_evidence'](reordered), slots)
    assert new[0] == 'strong' and reason == 'both'
    unknown = ns['align_candidate_evidence'](dict(evidence, candidate_evidence={}))
    assert gate.apply(base, keys, 0.2, unknown, slots)[1] is None
    assert gate.apply(base, keys, 0.95, fixed, slots)[1] is None
    assert evidence['S'] == 8.0, 'input evidence must not mutate'

    worker_ns = {}
    prefix = text(nb, 15).split("open('/kaggle/working/probe_core2.py'", 1)[0]
    exec(prefix, worker_ns)
    for name in ['CORE', 'V17_PATCH', 'RUNNER']:
        ast.parse(worker_ns[name])
    env = dict(np=np, os=SimpleNamespace(environ={}), time=time, PPM=10.0, PC_N1=5000, K=25,
               init_worker=lambda *args: None, _window=lambda target: ['strong', 'popular'])
    env['_W'] = dict(chem=SimpleNamespace(
        raw_fingerprint=lambda s: np.array([1000 if s == 'strong' else 500]), score_key=lambda s: s),
        Chem=SimpleNamespace(MolFromSmiles=lambda s: s),
        ecfp4=SimpleNamespace(GetFingerprintAsNumPy=lambda s: np.array([1])),
        mass=np.array([100.0, 100.0]), raw_e=np.array([0]), sel_e=np.array([0]), bits=np.array([0]),
        pool_keys=set(), ls=np.array([0.0, 100.0]), lp=np.array([0.0, 0.0]))
    exec(worker_ns['V17_PATCH'], env)
    env['_POP_LAM'], env['_POP_UNION'] = 0.25, 200
    _, smiles, _, keys, diag = env['probe_one'](('m', 100.0, np.array([1.0])))
    assert smiles[0] == 'popular' and diag['fz_top'] == 1000
    assert diag['candidate_evidence'][keys[0]]['raw_fz'] == 500
    assert diag['candidate_evidence']['strong']['popularity'] == 0


def check_expansion():
    nb = build_expansion()
    ns = dict(np=np, LIB_TAU=0.9)
    definitions(text(nb, 21), {'expansion_target', 'complete_forward_scores'}, ns)
    target, complete = ns['expansion_target'], ns['complete_forward_scores']
    assert target(['a'], 0.2, ['b'], True)
    for args in [(['a'], 0.95, ['b'], True), (['a'], 0.2, ['a'], True),
                 (['a'], 0.2, [], True), (['a'], 0.2, ['b'], False),
                 (['a'], float('nan'), ['b'], True)]:
        assert not target(*args)
    assert complete(['a', 'b'], dict(a=0.2, b=0.4))
    assert not complete(['a', 'b'], dict(a=0.2))
    assert not complete(['a', 'b'], dict(a=0.2, b=None))
    assert not complete(['a'], dict(a=float('nan')))
    assert not complete([], {})

    # Execute the actual collection/popularity loop with a deterministic fake engine.
    source = json.loads(SOURCE.read_text())
    count = 120
    C = SimpleNamespace(pid=np.arange(count), key=np.array([f'k{i}' for i in range(count)]),
                        smiles=np.array([f's{i}' for i in range(count)]), formula=np.array(['C'] * count))
    class Column:
        values = np.array(['[M+H]+'])
        def __eq__(self, value):
            return self.values == value
    class Frame:
        nm = SimpleNamespace(values=np.array([100.0]))
        adduct = Column()
        precursor_mz = SimpleNamespace(values=np.array([101.0]))
        def __setitem__(self, key, value):
            pass
        def groupby(self, *args, **kwargs):
            return [('m', self)]
        def itertuples(self):
            return []
    def loop(code):
        frame = Frame()
        scope = dict(np=np, pd=SimpleNamespace(read_parquet=lambda *a, **k: frame),
                     chem=SimpleNamespace(neutral_mass=lambda *a: np.array([100.0])),
                     os=__import__('os'), COMP='fake', P=SimpleNamespace(key=C.key),
                     find=lambda x: x, V=SimpleNamespace(run=lambda *a: (C, np.zeros((count, 0)),
                         np.zeros((count, 0)), [], {'lib_max': 0.2})), FEATURES=[],
                     rank_score=lambda *a: np.arange(count, 0, -1, dtype=float), TOPN=60,
                     LIB_TAU=0.9, ENG={'m': dict(keys=['different'])}, T0=time.time(), time=time)
        load = np.load
        try:
            np.load = lambda *a, **k: np.arange(count, dtype=float) / 50
            exec(code, scope)
        finally:
            np.load = load
        return scope
    old, new = loop(text(source, 21)), loop(text(nb, 21))
    assert old['BASE'] == new['BASE'], 'collecting 100 must not change baseline top60/popularity'
    assert len(new['EXP_FULL']['m']['smiles']) == 100
    assert new['EXP_SELECTED'] == ['m']
    assert new['EXP_FULL']['m']['keys'][:60] == old['BASE']['m'][1]
    assert new['EXP_FULL']['m']['keys'][60:] == list(C.key[60:100])
    assert text(nb, 15) == text(source, 15), 'second experiment must not include first fix'
    assert text(nb, 25) == text(source, 25), 'baseline gate stays unchanged'


def check_final_fallback():
    nb = build_expansion()
    cell = text(nb, 27)
    tail = cell[cell.index("baseline = pd.read_csv('submission.csv')"):]
    definitions_ns = dict(np=np, LIB_TAU=0.9, chem=SimpleNamespace(score_key=lambda s: s), ALPHA=0.6, KRR=3.0)
    definitions(cell, {'fuse2', '_form2'}, definitions_ns)
    definitions(text(nb, 21), {'complete_forward_scores'}, definitions_ns)
    baseline = pd.DataFrame(dict(molecule_id=['m', 'protected'], smiles=['s0;s1', 'locked']))
    class Forward:
        def __init__(self, mode):
            self.mode = mode
        def run_gl(self, package, items, **kwargs):
            assert kwargs['budget_s'] == 900
            if self.mode == 'timeout':
                raise TimeoutError('synthetic budget exhaustion')
            if self.mode == 'incomplete':
                return {'m': {'s0': 1.0}}
            return {'m': {s: float(i) for i, s in enumerate(items[0]['cands'])}}
        def rerank_multi(self, smiles, *args, **kwargs):
            return list(reversed(range(len(smiles))))
    def build_input(te, cands):
        return [dict(mid=m, cands=s, spectra=[dict(adduct='[M+H]+')]) for m, s in cands.items()]
    previous = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix='casmi-experiment-check-') as tmp:
            os.chdir(tmp)
            for mode in ['timeout', 'incomplete', 'complete']:
                baseline.to_csv('submission.csv', index=False)
                ns = dict(definitions_ns, pd=pd, json=json, os=os,
                    EXP_PRE_FUSION=baseline.copy(), EXP_SELECTED=['m'],
                    EXP_FULL={'m': dict(smiles=[f's{i}' for i in range(100)], keys=[f's{i}' for i in range(100)])},
                    ENG={'m': dict(smiles=['second'], keys=['second'])},
                    EXP_TOPN=100, EXP_GL_BUDGET=900,
                    te=pd.DataFrame(dict(molecule_id=['m'], adduct=['[M+H]+'])),
                    ice_fuse=SimpleNamespace(build_ice_input=build_input),
                    gl_fuse=Forward(mode), GL_PKG='mock', ICE_PKG='mock', ICE_LAM=1.0, GL_LAM=1.0,
                    ICE_SCORES={}, gc=SimpleNamespace(collect=lambda: None),
                    torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
                exec(tail, ns)
                actual = pd.read_csv('submission.csv')
                reference = pd.read_csv('submission_reference.csv')
                assert reference.equals(baseline)
                assert actual.iloc[1].smiles == 'locked'
                if mode != 'complete':
                    assert actual.equals(baseline), 'failed extra pass must preserve exact baseline'
                else:
                    assert 's99' in actual.iloc[0].smiles.split(';')
                    assert len(actual.iloc[0].smiles.split(';')) == 25
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    check_gate()
    check_expansion()
    check_final_fallback()
    for name in ['casmi26_candidate_aligned_confidence.ipynb', 'casmi26_targeted_candidate_expansion.ipynb']:
        nb = json.loads((ROOT / name).read_text())
        assert nb['nbformat'] == 4 and len(nb['cells']) == 28
        assert len(nb['metadata']['kaggle']['dataSources']) == 15
        for c in nb['cells']:
            assert c['cell_type'] == 'code' and c['execution_count'] is None and not c['outputs']
            ast.parse(''.join(c['source']))
    print('PASS: stale candidate regression, worker evidence, baseline top60 parity, full final-cell timeout/incomplete/success paths, tail candidate reaches top25, independent variants, notebook syntax.')
