'''Test faithful AST parity and append-only metric-safe mass coverage without model execution.'''
import ast
import json
import glob
import os
import sys
from unittest.mock import patch

import numpy as np

from build_flow_experiments import ROOT, validate
from build_metric_safe_adaptation import (SOURCE, WIDE_HELPER, SIMULATOR_WHEEL,
                                         compatible_simulator_setup, computational_ast)


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
                expected = ''.join(cell['source'])
                if 'def _claw_preseed_site(' in expected:
                    expected = compatible_simulator_setup(expected)
                actual = ''.join(nb['cells'][i+2]['source'])
                if i == 2:
                    actual = actual[:actual.index("'''Compatibility repair:")]
                assert computational_ast(expected) == computational_ast(actual)
                for node in ast.walk(ast.parse(''.join(cell['source']))):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str) and '\nimport ' in node.value:
                        ast.parse(node.value)
    assert adaptation['cells'][1:] == experiment['cells'][1:len(adaptation['cells'])]
    # Reproduce cp312 runtime with the incompatible cp313 wheel still attached.
    from pip._vendor.packaging.tags import Tag
    simulator = dict(glob=glob, os=os, sys=sys)
    selector = ast.parse(SIMULATOR_WHEEL).body[0]
    exec(ast.unparse(selector), simulator)
    choose = simulator['simulator_rdkit_wheel']
    bundled = '/pkg/wheels/rdkit-2025.3.6-cp312-cp312-manylinux_2_28_x86_64.whl.ice'
    external = '/input/rdkit-2025.3.6-cp313-cp313-manylinux_2_28_x86_64.whl'
    arm = '/input/rdkit-2025.3.6-cp312-cp312-manylinux_2_28_aarch64.whl'
    with patch('glob.glob', side_effect=[[bundled], [external, arm]]), patch(
            'pip._vendor.packaging.tags.sys_tags', return_value=iter([Tag('cp312', 'cp312', 'manylinux_2_28_x86_64')])):
        assert choose('/pkg') == bundled
    with patch('glob.glob', side_effect=[[bundled], [external]]), patch(
            'pip._vendor.packaging.tags.sys_tags', return_value=iter([Tag('cp313', 'cp313', 'manylinux_2_28_x86_64')])):
        assert choose('/pkg') == external
    with patch('glob.glob', side_effect=[[], [external, arm]]), patch(
            'pip._vendor.packaging.tags.sys_tags', return_value=iter([Tag('cp312', 'cp312', 'manylinux_2_28_x86_64')])):
        try:
            choose('/pkg')
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Do not select an incompatible wheel')
    # Exercise the actual patched installer without installing packages.
    installer_tree = ast.parse(''.join(adaptation['cells'][17]['source']))
    installer = next(n for n in installer_tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == '_claw_preseed_site')
    exec(ast.unparse(installer), simulator)
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    with TemporaryDirectory() as directory, patch('glob.glob', side_effect=[
            [bundled], [], [bundled], [external]]), patch('pip._vendor.packaging.tags.sys_tags',
            return_value=iter([Tag('cp312', 'cp312', 'manylinux_2_28_x86_64')])), patch(
            'shutil.copyfile'), patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='')) as pip:
        simulator['_claw_preseed_site']('/pkg', directory)
        command = pip.call_args.args[0]
        assert command[-1].endswith('cp312-cp312-manylinux_2_28_x86_64.whl'), command
        assert open(os.path.join(directory, '.ice_site_ok')).read() == os.path.basename(bundled)
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
    print('PASS: ranking AST parity, embedded Python, inputs, comment style, cp312/cp313 wheel selection, installer marker, metric-safe append and mass boundaries')
