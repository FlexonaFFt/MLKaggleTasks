'''Runnable synthetic checks; no Kaggle data, weights or network required.'''
import ast
import copy
import json
import sys
import types

import numpy as np

from build_flow_experiments import BASE, ROOT, DEFER_HELPER, source, validate


def function(value, name):
    tree = ast.parse(value)
    return ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


def embedded(nb):
    tree = ast.parse(source(nb, 15))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CORE' for t in n.targets))
    return ast.literal_eval(node.value)


def check_adduct(base, experiment):
    calls = []
    class Bank:
        def logits_el(self, items):
            calls.append(copy.deepcopy(items))
            return np.asarray([[x['prec'], x['mode'], x['adduct_ix'], x['n_merged']]
                               for x in items], dtype=np.float32), None

    chem = types.SimpleNamespace(adduct_index=lambda a: {'H': 1, 'Na': 2, 'neg': 3}[a])
    fpnet = types.SimpleNamespace(prep_peaks=lambda mz, it, prec: (mz, it))
    casmi = types.ModuleType('casmi')
    casmi.chem, casmi.fpnet = chem, fpnet
    spectra_module = types.ModuleType('casmi.spectra')
    spectra_module.merge_spectra = lambda spectra: (np.concatenate([x[0] for x in spectra]),
                                                   np.concatenate([x[1] for x in spectra]))
    previous = {key: sys.modules.get(key) for key in ('casmi', 'casmi.spectra')}
    sys.modules.update({'casmi': casmi, 'casmi.spectra': spectra_module})
    try:
        functions = []
        for nb in (base, experiment):
            namespace = {'np': np}
            exec(function(embedded(nb), 'molecule_logits'), namespace)
            functions.append(namespace['molecule_logits'])
        def sp(ad='H', mode=1, prec=101.0, ce=20):
            return dict(adduct=ad, mode=mode, prec=prec, ce=ce, ce_n=1,
                        mz=np.asarray([40.0]), it=np.asarray([1.0]))
        for data in ([], [sp()], [sp(), sp(prec=103, ce=None)],
                     [sp(), sp('neg', -1, 99)], [sp('neg', -1, 99, None)]):
            old, new = [f(Bank(), data) for f in functions]
            if old is None:
                assert new is None
            else:
                assert np.array_equal(old, new), (old, new)
        data = [sp(), sp('Na', 1, 123), sp('neg', -1, 99)]
        calls.clear()
        new = functions[1](Bank(), data)
        singles, merged = calls
        assert len(singles) == 3 and len(merged) == 3
        assert [(s['prec'], s['adduct_ix']) for s in merged] == [(101, 1), (123, 2), (99, 3)]
        zs = np.asarray([[s['prec'], s['mode'], s['adduct_ix'], s['n_merged']] for s in singles], dtype=np.float32)
        zm = np.asarray([[s['prec'], s['mode'], s['adduct_ix'], s['n_merged']] for s in merged], dtype=np.float32)
        expected = .5 * (zs.mean(0) + np.asarray([zm[:2].mean(0), zm[2]]).mean(0))
        assert np.array_equal(new, expected), 'Polarities must retain equal weight'
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def check_defer(base, experiment):
    namespace = {}
    exec(DEFER_HELPER, namespace)
    defer = namespace['defer_pubchem_glacier']
    assert defer(1, {'a': None}, {'a': .8}, {'1': {'smiles': ['a']}})
    for ice, gl, engines in [({}, {'a': .8}, {'1': {'smiles': ['a']}}),
                              ({'a': None}, {}, {'1': {'smiles': ['a']}}),
                              ({'a': None}, {'a': .8}, {}),
                              ({'a': None}, {'a': .8}, {'1': {'smiles': []}})]:
        assert not defer(1, ice, gl, engines)
    pc = {m: dict(pc=['a', 'b'], pc_keys=['ka', 'kb'], pc_fz=[10., 5.], pc_form=['F', 'F'])
          for m in (1, 2, 3)}
    common = dict(BASE={m: (['x'], ['kx'], .1, [1.], ['X']) for m in pc},
                  PC=pc, ICE_PC=True, BASE_ICE=True, ICE_CHUNKS=0, TOPN=60, ICE_LAM=.5,
                  ICE_SCORES={str(m): {'a': None, 'b': None} for m in pc},
                  ENG={'1': {'smiles': ['x']}, '2': {'smiles': ['x']}},
                  gl_of=lambda mid: {'a': .8} if mid in (1, 3) else {},
                  gl_rerank=lambda order, smis, keys, scs, forms, ice, gl, *args: [1, 0] if gl else order,
                  ice_fuse=types.SimpleNamespace(rerank=lambda *args, **kwargs: [0, 1]),
                  ice_stats={}, gl_stats={}, T0=0, time=types.SimpleNamespace(time=lambda: 0))
    original, modified = copy.deepcopy(common), copy.deepcopy(common)
    old = source(base, 23)
    old = old[old.index('for m in list(BASE):'):]
    exec(old, original)
    new = source(experiment, 23)
    new = new[new.index('import copy\nPC_BASELINE_POST'):]
    exec(new, modified)
    assert modified['PC_GL_DEFERRED'] == {'1'}
    assert modified['PC'][1] == pc[1], 'Eligible PubChem order must be restored before gate'
    assert modified['PC_BASELINE_POST'] == original['PC'], 'Baseline snapshot must retain original early reranking'
    assert all(modified['PC'][m] == original['PC'][m] for m in (2, 3)), 'No-GL and no-engine paths must be unchanged'
    assert modified['BASE'] == original['BASE']


if __name__ == '__main__':
    base = json.loads(BASE.read_text())
    names = ['casmi26_0420_single_pass_glacier.ipynb', 'casmi26_0420_adduct_safe_pubchem.ipynb']
    single, adduct = [json.loads((ROOT / 'experiments' / name).read_text()) for name in names]
    for nb, allowed in [(single, {0, 1, 2, 3, 5, 6, 23, 27}),
                        (adduct, {0, 1, 2, 3, 5, 6, 15, 27})]:
        validate(nb)
        assert nb['metadata']['kaggle'] == base['metadata']['kaggle'], 'Input metadata changed'
        assert len(nb['cells']) == len(base['cells'])
        assert all(source(nb, i) == source(base, i) for i in range(len(nb['cells'])) if i not in allowed)
        assert all(not c.get('outputs') and c.get('execution_count') is None
                   for c in nb['cells'] if c['cell_type'] == 'code')
    check_adduct(base, adduct)
    check_defer(base, single)
    print('PASS: syntax, embedded modules, isolated edits, input metadata, adduct parity/weighting and GL deferral/fallbacks')
