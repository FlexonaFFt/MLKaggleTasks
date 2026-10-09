'''Test faithful AST parity and append-only metric-safe mass coverage without model execution.'''
import ast
import json

import numpy as np

from build_flow_experiments import ROOT, validate
from build_metric_safe_adaptation import SOURCE, WIDE_HELPER, computational_ast


if __name__ == '__main__':
    original = json.loads(SOURCE.read_text())
    adaptation = json.loads((ROOT / 'experiments/casmi26_0433_metric_safe_adaptation.ipynb').read_text())
    experiment = json.loads((ROOT / 'experiments/casmi26_0433_safe_wide_mass_tail.ipynb').read_text())
    for nb in (adaptation, experiment):
        validate(nb)
        assert nb['metadata']['kaggle'] == original['metadata']['kaggle']
        assert all(c['cell_type'] == 'code' and c['execution_count'] is None and c['outputs'] == [] for c in nb['cells'])
        for i, cell in enumerate(original['cells']):
            if cell['cell_type'] == 'code':
                assert computational_ast(''.join(cell['source'])) == computational_ast(''.join(nb['cells'][i+2]['source']))
                for node in ast.walk(ast.parse(''.join(cell['source']))):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str) and '\nimport ' in node.value:
                        ast.parse(node.value)
    assert adaptation['cells'][1:] == experiment['cells'][1:len(adaptation['cells'])]
    # Synthetic identities test the inherited control flow, not real RDKit canonicalization.
    namespace = dict(np=np, metric_identity=lambda s: None if s == 'INVALID' else s.removesuffix('_alias'))
    tree = ast.parse(''.join(original['cells'][21]['source']))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('safe_extend', 'first_ranks', 'assert_monotone'):
            exec(ast.unparse(node), namespace)
    exec(WIDE_HELPER, namespace)
    extend = namespace['safe_extend']
    assert extend(['a', 'a_alias', 'INVALID', 'b'], ['b', 'c']) == ['a', 'b', 'c']
    namespace['assert_monotone'](['a', 'a_alias', 'INVALID', 'b'], ['a', 'b', 'c'])
    mass = np.asarray([100.003, 100.0015, 100.0, 99.998, 100.002])
    smiles = np.asarray(['outside', 'new', 'a_alias', 'INVALID', b'boundary'], dtype=object)
    order = np.argsort(mass, kind='stable')
    result = namespace['wide_mass_tail'](['a', 'b'], 100., mass, smiles, order)
    assert result == ['a', 'b', 'new', 'boundary'], result
    full = [f'x{i}' for i in range(25)]
    assert namespace['wide_mass_tail'](full, 100., mass, smiles, order) == full
    assert namespace['wide_mass_tail'](['a'], np.nan, mass, smiles, order) == ['a']
    assert namespace['wide_mass_tail'](['a'], 0., mass, smiles, order) == ['a']
    assert namespace['wide_mass_tail'](['a'], 300., mass, smiles, order) == ['a']
    assert namespace['wide_mass_tail'](['a'], 100., np.asarray([]), np.asarray([]), np.asarray([], dtype=int)) == ['a']
    print('PASS: faithful computational ASTs, embedded Python, inherited inputs, comment style, metric-safe append logic and mass boundaries')
