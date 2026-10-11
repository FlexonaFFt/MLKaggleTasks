'''AST parity plus GL-present, ICE-only, empty evidence and expanded-window regression tests.'''
import ast
import json
from types import SimpleNamespace
from build_ice_fallback_versions import SOURCE, build
from build_metric_safe_adaptation import computational_ast
from build_aggressive_fusion_experiments import METRIC_HELPER

original = json.loads(SOURCE.read_text())
faithful, wide = build()
for i, cell in enumerate(original['cells']):
    if cell['cell_type'] == 'code':
        assert computational_ast(''.join(cell['source'])) == computational_ast(''.join(faithful['cells'][i+2]['source']))
assert faithful['cells'][1:] == wide['cells'][1:len(faithful['cells'])]
assert faithful['metadata']['kaggle'] == wide['metadata']['kaggle'] == original['metadata']['kaggle']
assert all(c['cell_type'] == 'code' and c['execution_count'] is None and not c['outputs'] for c in wide['cells'])

tree = ast.parse(''.join(original['cells'][-1]['source']))
body = tree.body[0].body
functions = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in ('_finite_coverage', 'rerank_row')]
env = {}
exec('import math\n' + METRIC_HELPER, env)
exec(ast.unparse(ast.Module(body=functions, type_ignores=[])), env)
rerank = env['rerank_row']
calls = []
def ice_rank(candidates, keys, scores, formulas, evidence, **kwargs):
    calls.append('ice')
    return list(reversed(range(len(candidates)))) if any(v is not None for v in evidence.values()) else list(range(len(candidates)))
def gl_rank(candidates, keys, scores, formulas, evidence, weights, **kwargs):
    calls.append('gl')
    assert evidence[0] == {}
    return list(range(len(candidates)))
def extend(first, extra, limit=25):
    return list(dict.fromkeys(first + extra))[:limit]
scope = dict(ENG={'m': {'smiles': [f'E{i}' for i in range(40)]}},
    PREFUSION_LISTS={'m': [f'A{i}' for i in range(25)]},
    metric_fusion=env['metric_fusion'], metric_identity=lambda s: s,
    chem=SimpleNamespace(score_key=lambda s: s), ALPHA=.6, KRR=3., ICE_LAM=1., GL_LAM=1.,
    ICE_SCORES={'m': {'A0': 1., 'E39': 2.}}, gl_of=lambda mid: {}, _form2=lambda s: 'F',
    ice_fuse=SimpleNamespace(rerank=ice_rank), gl_fuse=SimpleNamespace(rerank_multi=gl_rank), safe_extend=extend)
result40, audit40 = rerank(scope, 'm', ['A0'], window=40, ice_fallback=True, processed=True)
assert calls == ['ice'] and audit40['branch'] == 'ice_fallback'
result80, audit80 = rerank(scope, 'm', ['A0'], window=80, ice_fallback=True, processed=True)
assert audit40['candidate_count'] == 40 and audit80['candidate_count'] == 65
assert result40 != result80 and len(result80) == 25
calls.clear()
scope['gl_of'] = lambda mid: {'A0': 2.}
rerank(scope, 'm', ['A0'], window=40, ice_fallback=True, processed=True)
assert calls == ['gl']
scope['gl_of'] = lambda mid: {}
scope['ICE_SCORES'] = {'m': {'A0': None}}
yes, audit = rerank(scope, 'm', ['A0'], window=40, ice_fallback=True, processed=True)
no, _ = rerank(scope, 'm', ['A0'], window=40, ice_fallback=False, processed=True)
assert yes == no and audit['ice_finite_count'] == 0
scope['ICE_SCORES'] = {}
calls.clear()
rerank(scope, 'm', ['A0'], window=40, ice_fallback=True, processed=True)
assert calls == []
assert rerank(scope, 'm', ['placeholder'], window=80, ice_fallback=True, processed=False)[0] == ['placeholder']
print('PASS: exact uploaded-code AST, unchanged inputs, independent window factor, fallback gates, placeholders')
