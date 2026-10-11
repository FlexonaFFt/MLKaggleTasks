'''Synthetic evidence boundaries plus complete paired-overlay execution without model inference.'''
import ast
import hashlib
import json
import os
import tempfile
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from build_aggressive_fusion_experiments import METRIC_HELPER
from build_ice_fallback_versions import build as e006_build, SOURCE
from build_simulator_evidence_experiments import build, HELPERS
from build_flow_experiments import ROOT

env = {}
exec(HELPERS + METRIC_HELPER, env)
identity = lambda s: None if s == 'invalid' else s.split('-')[0]
formula = lambda s: 'OTHER' if s.endswith('-charge') else 'F'
align = env['aligned_simulator_scores']
raw = {'B-alt': 2., 'B-other': 4., 'A': 9., 'C-charge': 8., 'invalid': 100.}
mapped, n = align(raw, ['A', 'B', 'C'], ['ka', 'kb', 'kc'], list(raw), identity, formula, lambda s: 'k'+s)
assert mapped['A'] == 9. and mapped['B'] == 3. and n == 1 and 'C' not in mapped
assert raw == {'B-alt': 2., 'B-other': 4., 'A': 9., 'C-charge': 8., 'invalid': 100.}
mapped, n = align({'B': float('nan'), 'B-alt': 0.}, ['B'], ['kb'], ['B-alt'], identity, formula, lambda s:s)
assert mapped['B'] == 0. and n == 1
joint = env['complete_group_ice']
candidates, keys = ['A', 'B', 'C'], ['ka', 'kb', 'kc']
accepted, n = joint(candidates, keys, ['F', 'F', 'G'], {'A': 1., 'B': 2., 'C': 3.}, {'A': 3., 'B': 2., 'C': 1.})
assert accepted == {'A': 1., 'B': 2.} and n == 1
for missing in ({'A': 1.}, {'A': 1., 'B': None}, {'A': 1., 'B': float('inf')}):
    assert joint(['A','B'], ['ka','kb'], ['F','F'], missing, {'A':2.,'B':1.}) == ({}, 0)
# Exercise the real public reranker: excluded formula groups must retain their GL-only order.
spec = importlib.util.spec_from_file_location('real_gl_fuse', ROOT / 'archive/simulator-interface-2026-10-09/gl_fuse.py')
real_gl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_gl)
smis = ['A','B','C','D','E']
forms = ['F','F','G','G','G']
ice = {'A':3.,'B':1.,'C':1.,'D':3.}  # E is missing: group G must be excluded.
gl = {'A':1.,'B':3.,'C':1.,'D':2.,'E':3.}
selected, _ = joint(smis, smis, forms, ice, gl)
original_order = real_gl.rerank_multi(smis, smis, [5.,4.,3.,2.,1.], forms, [{},gl], [1.,1.])
joint_order = real_gl.rerank_multi(smis, smis, [5.,4.,3.,2.,1.], forms, [selected,gl], [1.,1.])
assert joint_order[2:] == original_order[2:]

base = e006_build()[0]
notebooks = build()
for nb in notebooks:
    assert nb['cells'][1:len(base['cells'])] == base['cells'][1:]
    assert nb['metadata']['kaggle'] == base['metadata']['kaggle']
    assert all(c['execution_count'] is None and not c['outputs'] for c in nb['cells'])

# Use the original pure row reconstruction as the control; mocks provide simulator ordering only.
original = json.loads(SOURCE.read_text())
tree = ast.parse(''.join(original['cells'][-1]['source']))
funcs = [n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name in ('_finite_coverage', 'rerank_row')]
exec('import math\n' + ast.unparse(ast.Module(body=funcs, type_ignores=[])), env)
def extend(first, extra, limit=25):
    return list(dict.fromkeys(first + extra))[:limit]
def gl_order(candidates, keys, scores, formulas, columns, lams, **kwargs):
    # Change only if the experimental ICE column was enabled or an alias filled B.
    flip = bool(columns[0]) or columns[1].get('B') == 99.
    return list(reversed(range(len(candidates)))) if flip else list(range(len(candidates)))
def ice_order(candidates, *args, **kwargs):
    return list(range(len(candidates)))
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as directory:
    try:
        os.chdir(directory)
        for i, notebook in enumerate(notebooks):
            gl = {'A':1., 'B-alt':99., 'C':3.} if i == 0 else {'A':1., 'B':2., 'C':3.}
            scope = dict(env, pd=pd, json=json, Path=Path,
                sha256=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                ENG={'m': {'smiles':['B','C']}}, PREFUSION_LISTS={'m':['A','B','C']},
                metric_identity=identity, metric_fusion=env['metric_fusion'],
                BASE={'m': ([], [], .1)}, BASE_TAIL_CANDIDATES={'m':['B-alt']}, PC={}, LIB_TAU=.9,
                ICE_SCORES={'m':{'A':3.,'B':2.,'C':1.}}, gl_of=lambda mid:gl,
                chem=SimpleNamespace(score_key=lambda s:s), _form2=formula,
                ALPHA=.6, KRR=3., ICE_LAM=1., GL_LAM=1., safe_extend=extend,
                gl_fuse=SimpleNamespace(rerank_multi=gl_order), ice_fuse=SimpleNamespace(rerank=ice_order),
                te=pd.DataFrame({'molecule_id':['m']}))
            reference = ['A','B','C']
            control, _ = env['rerank_row'](scope, 'm', reference, window=40, ice_fallback=False, processed=True)
            e006, _ = env['rerank_row'](scope, 'm', reference, window=40, ice_fallback=True, processed=True)
            def reset():
                pd.DataFrame([('m',';'.join(control)), ('dummy','CCO')], columns=['molecule_id','smiles']).to_csv('submission_e004_control.csv',index=False)
                pd.DataFrame([('m',';'.join(reference)), ('dummy','CCO')], columns=['molecule_id','smiles']).to_csv('submission_reference.csv',index=False)
                pd.DataFrame([('m',';'.join(e006)), ('dummy','CCO')], columns=['molecule_id','smiles']).to_csv('submission.csv',index=False)
                Path('e004_verification.json').write_text(json.dumps(dict(experiment='E004', processed_ids=['m'],
                    hashes={'submission.csv':scope['sha256']('submission_e004_control.csv')})))
                Path('experiment_verification.json').write_text('{}')
                Path('experiment_audit_factorial.csv').write_text('molecule_id\nm\n')
            reset()
            expected = Path('submission.csv').read_bytes()
            exec(''.join(notebook['cells'][-1]['source']), scope)
            assert Path('submission_e006_control.csv').read_bytes() == expected
            report = json.loads(Path('paired_e006_audit.json').read_text())
            assert report['changed_rows'] == 1
            assert pd.read_csv('submission.csv').iloc[1].smiles == 'CCO'
            if i == 1:
                scope['BASE']['m'] = ([],[],.95)
            else:
                scope['BASE_TAIL_CANDIDATES'] = {}
            reset()
            exec(''.join(notebook['cells'][-1]['source']), scope)
            assert json.loads(Path('paired_e006_audit.json').read_text())['changed_rows'] == 0
    finally:
        os.chdir(cwd)
print('PASS: score alias means/direct preservation/formula guard, complete coverage, strong-library gate, paired overlays and placeholders')
