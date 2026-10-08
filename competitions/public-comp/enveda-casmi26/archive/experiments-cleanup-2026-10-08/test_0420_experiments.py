"""Local regression checks for the two 0.420-based hypotheses."""
import ast
import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from build_0420_experiments import BASE, shield, stronger_engine
from build_candidate_experiments import text


def check():
    base = json.loads(BASE.read_text())
    guarded, strong = shield(), stronger_engine()
    for nb in [guarded, strong]:
        assert nb['metadata']['kaggle'] == base['metadata']['kaggle']
        for i in [9, 11, 15, 17, 19, 21, 23, 25]:
            assert text(nb, i) == text(base, i), ('unexpected baseline change', i)
        for cell in nb['cells']:
            assert cell['cell_type'] == 'code'
            ast.parse(''.join(cell['source']))
    assert text(guarded, 13) == text(base, 13)
    assert text(strong, 27).startswith(text(base, 27))

    # Validate the actual subprocess script rather than the outer string alone.
    tree = ast.parse(text(strong, 13))
    scripts = [node.args[0].value for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'write' and node.args and isinstance(node.args[0], ast.Constant)
               and isinstance(node.args[0].value, str)]
    runner = next(s for s in scripts if 'E.RANK.SEEDS' in s)
    ast.parse(runner)
    assert 'E.RANK.SEEDS = (0, 1, 2, 3); E.CFG.N_ANALOG = 250' in runner
    assert 'EXPERIMENT ranker seeds' in runner
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for code in ast.literal_eval(node.value).values():
                ast.parse(code)

    ns = dict(np=np, LIB_TAU=0.9, chem=SimpleNamespace(score_key=lambda s: s.split(':')[0]))
    tree = ast.parse(text(guarded, 27))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'protect_final_top1')
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<shield>', 'exec'), ns)
    protect = ns['protect_final_top1']
    assert protect(['b', 'a'], ['a'], 0.95, [])[0] == ['a', 'b']
    assert protect(['b', 'a'], ['a'], 0.1, ['a'])[0] == ['a', 'b']
    assert protect(['b', 'a'], ['a'], 0.1, ['c']) == (['b', 'a'], None)
    assert protect(['b'], [], 0.95, ['a']) == (['b'], None)
    assert protect(['b'], ['a'], float('nan'), ['c']) == (['b'], None)
    assert protect(['a:other', 'b'], ['a:original'], 0.95, [])[0] == ['a:other', 'b']
    assert len(protect([str(i) for i in range(25)], ['new'], 0.95, [])[0]) == 25

    # Execute final postprocessing on real CSVs, covering protected/unprotected rows.
    source = text(guarded, 27)
    tail = source[source.index("os.replace('submission_reference.csv'"):]
    frame = pd.DataFrame(dict(molecule_id=['lib', 'consensus', 'none'], smiles=['b;a', 'd;c', 'f;e']))
    before = pd.DataFrame(dict(molecule_id=['lib', 'consensus', 'none'], smiles=['a;b', 'c;d', 'e;f']))
    previous = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix='casmi-top1-check-') as tmp:
            os.chdir(tmp)
            frame.to_csv('submission.csv', index=False)
            frame.to_csv('submission_reference.csv', index=False)
            with open('experiment_diagnostics.json', 'w') as f:
                json.dump({'experiment': 'candidate_aligned_gate'}, f)
            ns.update(pd=pd, os=os, json=json, SHIELD_PRE_FUSION=before,
                      BASE={'lib': ([], [], 0.95), 'consensus': ([], [], 0.1), 'none': ([], [], 0.1)},
                      ENG={'consensus': {'keys': ['c']}, 'none': {'keys': ['x']}})
            exec(tail, ns)
            actual = pd.read_csv('submission.csv')
            assert actual.smiles.tolist() == ['a;b', 'c;d', 'f;e']
            assert pd.read_csv('submission_reference.csv').equals(frame)
            report = json.load(open('experiment_diagnostics.json'))
            assert report['final_lists_changed'] == 2
    finally:
        os.chdir(previous)
    print('PASS: final shields, unchanged rows, canonical consensus, top25 bound, baseline isolation, embedded runner settings/syntax, unchanged input metadata.')


if __name__ == '__main__':
    check()
