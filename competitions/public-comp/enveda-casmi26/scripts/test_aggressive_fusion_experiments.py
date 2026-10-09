'''Focused tests: independent baseline parity, unique votes, coverage fallback and runner contract.'''
import ast
import json
import math
import importlib.util
import tempfile
import os
import hashlib
from types import SimpleNamespace
import pandas as pd
from build_aggressive_fusion_experiments import GLOBAL_RUN, METRIC_RUN
from build_aggressive_fusion_experiments import build, METRIC_HELPER, GLOBAL_HELPER
from build_metric_safe_adaptation import build as baseline, computational_ast
from build_flow_experiments import ROOT, validate


def test():
    reference = baseline()[0]
    notebooks = build()
    for notebook in notebooks:
        validate(notebook)
        assert notebook['metadata']['kaggle'] == reference['metadata']['kaggle']
        # No baseline arithmetic changes: only one read-only snapshot before fusion.
        for expected, actual in zip(reference['cells'][2:], notebook['cells'][3:-1]):
            value = ''.join(actual['source'])
            value = '\n'.join(line for line in value.split('\n') if not line.startswith('PREFUSION_LISTS = '))
            assert computational_ast(value) == computational_ast(''.join(expected['source']))
        assert all(cell['cell_type'] == 'code' and not cell['outputs'] for cell in notebook['cells'])
    env = {}
    exec(METRIC_HELPER + GLOBAL_HELPER, env)
    identity = lambda s: None if s == 'invalid' else {'A-alt': 'A'}.get(s, s)
    fused, scores, collapsed = env['metric_fusion'](['A', 'A-alt', 'invalid', 'B'], ['B', 'B', 'C'], identity)
    assert fused == ['B', 'A', 'C'] and collapsed == 2
    assert math.isclose(scores[0], 1 / 5 + 0.6 / 4)
    assert env['metric_fusion'](['A'], ['A-alt'], identity)[1] == [1.6 / 4]
    rank = env['global_glacier_order']
    assert rank(['A', 'B', 'C'], {'A': 0., 'B': 0., 'C': 100.}, weight=2.)[0] == 2
    for evidence in ({}, {'A': 1.}, {'A': 1., 'B': None},
                     {'A': 1., 'B': float('nan')}, {'A': 1., 'B': 1.}, {'A': 1., 'B': '2'}):
        assert rank(['A', 'B'], evidence) is None
    assert rank([], {}) is None
    # The real public helper accepts singleton-formula candidates when ice_candidates is bypassed.
    spec = importlib.util.spec_from_file_location('public_gl_fuse', ROOT / 'archive/simulator-interface-2026-10-09/gl_fuse.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    same = module.rerank_multi(['A', 'B', 'C'], ['a', 'b', 'c'], [3., 2., 1.],
                             ['F1', 'F2', 'F3'], [{'A': 0., 'B': 0., 'C': 100.}], [2.], top_n=3)
    assert same == [0, 1, 2]  # Confirms the structural restriction the global experiment tests.
    for nb in notebooks:
        ast.parse(''.join(nb['cells'][-1]['source']))
    # Execute both complete overlays against a tiny fixture; no model inference or Kaggle files.
    def extend(original, extra, limit=25):
        return list(dict.fromkeys(list(original) + list(extra)))[:limit]
    def run_gl(pkg, items, **kwargs):
        assert kwargs['workdir'].endswith('gl_global_work')
        assert len(items) == 1 and items[0]['mid'] == 'hard'
        assert items[0]['cands'] == ['A', 'B', 'C']
        return {'hard': {'A': 0., 'B': 0., 'C': 100.}}
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as folder:
        try:
            os.chdir(folder)
            reference_frame = pd.DataFrame([('hard', 'A;B'), ('strong', 'A;B'), ('unsupported', 'A;B')],
                                           columns=['molecule_id', 'smiles'])
            fixture = dict(pd=pd, json=json, os=os, sha256=lambda path: hashlib.sha256(open(path, 'rb').read()).hexdigest(),
                chem=SimpleNamespace(score_key=identity), metric_identity=identity, safe_extend=extend,
                ALPHA=.6, KRR=3., LIB_TAU=.9, ICE_LAM=.5, GL_LAM=2.,
                GL_BUDGET=300, GL_PKG='/gl', ICE_PKG='/ice', _form2=lambda s: s,
                BASE={'hard': ([], [], .2), 'strong': ([], [], .99), 'unsupported': ([], [], .1)}, PC={},
                ENG={'hard': {'smiles': ['B', 'C']}}, PREFUSION_LISTS={'hard': ['A', 'A-alt', 'B']},
                ICE_SCORES={}, gl_of=lambda mid: {}, FUSION_OVERFLOW={}, BASE_TAIL_CANDIDATES={'hard': ['C']},
                te=pd.DataFrame([('hard', 'positive', '[M+H]+'), ('strong', 'positive', '[M+H]+'),
                                 ('unsupported', 'negative', '[M-H]-')],
                                columns=['molecule_id', 'ionization_mode', 'adduct']),
                ice_fuse=SimpleNamespace(build_ice_input=lambda frame, candidates:
                    [dict(mid=mid, cands=cands, spectra=[dict(adduct='[M+H]+')]) for mid, cands in candidates.items()]),
                gl_fuse=SimpleNamespace(run_gl=run_gl))
            for run in (METRIC_RUN, GLOBAL_RUN):
                reference_frame.to_csv('submission.csv', index=False)
                exec(run, dict(fixture))
                output = pd.read_csv('submission.csv')
                assert output.iloc[0].smiles.startswith('B' if run == METRIC_RUN else 'C')
                assert output.iloc[1:].equals(reference_frame.iloc[1:])
                assert pd.read_csv('submission_reference.csv').equals(reference_frame)
            fixture['gl_fuse'] = SimpleNamespace(run_gl=lambda *args, **kwargs: {'hard': {'A': 1.}})
            reference_frame.to_csv('submission.csv', index=False)
            exec(GLOBAL_RUN, dict(fixture))
            assert pd.read_csv('submission.csv').equals(reference_frame)
        finally:
            os.chdir(cwd)
    print('PASS: source parity, independent overlays, metric votes, global ranking, incomplete-score fallback')


if __name__ == '__main__':
    test()
