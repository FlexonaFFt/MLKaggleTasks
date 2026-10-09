'''Preserve the public V1 source and build a faithful adaptation plus an append-only experiment.'''
import ast
import copy
import hashlib
import io
import json
import shutil
import tokenize
from pathlib import Path

from build_flow_experiments import ROOT, prose, validate

DOWNLOAD = Path('/Users/flexonafft/Downloads/casmi26-v1-1-metric-audit-and-safe-tail.ipynb')
SOURCE = ROOT / 'archive/public-sources-2026-10-09/obstacledeveloper_v1_356393001.ipynb'
URL = 'https://www.kaggle.com/code/obstacledeveloper/casmi26-v1-1-metric-audit-and-safe-tail?scriptVersionId=356393001'


def comments(value):
    lines = value.splitlines(keepends=True)
    top = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(value).readline)
           if t.type == tokenize.COMMENT and t.start[1] == 0}
    for line in top:
        content = lines[line - 1].lstrip('#').strip()
        if "'''" not in content:
            lines[line - 1] = prose(content)
    ast.parse(''.join(lines))
    return ''.join(lines)


def code_cell(value):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                source=value.splitlines(keepends=True))


def computational_ast(value):
    class WithoutNarrative(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)
    return ast.dump(WithoutNarrative().visit(ast.parse(value)), include_attributes=False)


WIDE_HELPER = '''
def wide_mass_tail(original, target, masses, smiles, order):
    if len(original) >= 25 or not np.isfinite(target) or target <= 0:
        return list(original)
    ordered_mass = masses[order]
    lo = np.searchsorted(ordered_mass, target * (1 - 20e-6), side='left')
    hi = np.searchsorted(ordered_mass, target * (1 + 20e-6), side='right')
    indices = order[lo:hi]
    indices = indices[np.argsort(np.abs(masses[indices] - target), kind='stable')]
    extra = [s.decode() if isinstance(s, bytes) else str(s) for s in smiles[indices]]
    revised = safe_extend(original, extra)
    assert_monotone(original, revised)
    assert revised[:len(original)] == list(original), 'Never reorder the completed V1.1 list'
    return revised
'''


WIDE_RUN = prose('Experimental overlay: 20 ppm vacant-slot coverage\n'
    'Run the entire faithful V1.1 pipeline first. Only lists still shorter than 25 receive '
    'additional DreaMS structures from a 20 ppm mass window, nearest mass first. '
    'This is structure retrieval, not DreaMS embedding inference. Never replace an existing candidate. '
    'Missing DreaMS data, no vacancies or no novel valid structures makes this experiment a no-op. '
    'Its ceiling is low when almost all baseline lists are already full.') + WIDE_HELPER + '''
_faithful = pd.read_csv('submission.csv')
_faithful.to_csv('submission_reference.csv', index=False)
_wide_audit, _wide_rows = [], []
for _row in _faithful.itertuples(index=False):
    _mid = str(_row.molecule_id)
    _original = str(_row.smiles).split(';')
    _revised = list(_original)
    _target = float(_targets.get(_row.molecule_id, _targets.get(_mid, np.nan)))
    if _mid in _active and len(_original) < 25 and np.isfinite(_target) and _target > 0 and _dm_paths:
        if _dm_mass is None:
            _directory = Path(_dm_paths[0]).parent
            _dm_mass = np.load(_directory / 'dreams_np_mass.npy').astype(np.float64)
            _dm_smiles = np.load(_directory / 'dreams_np_smiles.npy', allow_pickle=True)
            assert len(_dm_mass) == len(_dm_smiles)
            _dm_order = np.argsort(_dm_mass, kind='stable')
        _revised = wide_mass_tail(_original, _target, _dm_mass, _dm_smiles, _dm_order)
    assert _revised[:len(_original)] == _original
    if _mid in _active:
        assert_monotone(_original, _revised)
        assert len(first_ranks(_revised)) == len(_revised)
    _wide_rows.append((_row.molecule_id, ';'.join(_revised)))
    _wide_audit.append(dict(molecule_id=_mid, active=_mid in _active,
                            vacant_before=max(0, 25-len(_original)), added=len(_revised)-len(_original),
                            changed=_revised != _original))
_wide_final = pd.DataFrame(_wide_rows, columns=['molecule_id', 'smiles'])
assert _wide_final.molecule_id.equals(_faithful.molecule_id) and _wide_final.molecule_id.is_unique
assert _wide_final.smiles.notna().all() and _wide_final.smiles.str.split(';').map(len).between(1, 25).all()
_wide_final.to_csv('submission.csv', index=False)
_wide_changed = sum(r['changed'] for r in _wide_audit)
pd.DataFrame(_wide_audit).to_csv('experiment_audit.csv', index=False)
Path('experiment_diagnostics.json').write_text(json.dumps(dict(
    experiment='metric_safe_wide_mass_tail', source_public_score=0.433, measured_score=None,
    reference_scope='Completed faithful V1.1 from the same run, not historical score reproduction',
    ppm=20.0, changed_molecules=_wide_changed, added=sum(r['added'] for r in _wide_audit),
    active_vacant_molecules=sum(r['active'] and r['vacant_before'] > 0 for r in _wide_audit),
    dreams_available=bool(_dm_paths), monotone_assertions_passed=True,
    same_as_reference=_wide_changed == 0,
    reference_sha256=hashlib.sha256(Path('submission_reference.csv').read_bytes()).hexdigest(),
    submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()), indent=2))
print('Wide mass tail changed molecules:', _wide_changed)
if _wide_changed == 0:
    print('DO NOT SPEND A SECOND SUBMISSION: experimental CSV equals faithful adaptation')
'''


def build():
    source = json.loads(SOURCE.read_text())
    nb = copy.deepcopy(source)
    nb['metadata'].pop('papermill', None)
    for cell in nb['cells']:
        value = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            cell.update(code_cell(prose(value.replace('\\', '\\\\'))))
        else:
            rewritten = comments(value)
            assert computational_ast(value) == computational_ast(rewritten)
            cell.update(code_cell(rewritten))
    credits = prose('Credits & Attribution\n'
        '@obstacledeveloper — direct source: CASMI26 V1.1 Metric Audit and Safe Tail, version 1, '
        'scriptVersionId 356393001. Source: ' + URL + '\n'
        'The source page displays public score 0.433; this adaptation has not reproduced that result.\n'
        '@bobthebot369 — inherited Sovereign Zenith ensemble and confidence-promotion ideas: '
        'https://www.kaggle.com/code/bobthebot369/enveda-casmi-2026-v18-sovereign-zenith/notebook\n'
        '@ahmedberatozer — engine, ranker, pool, PubChem, full FPNet and simulator packages: '
        'https://www.kaggle.com/ahmedberatozer\n'
        '@prvsiyan — analog propagation, spectral matching and fingerprint models: '
        'https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline\n'
        '@dmitriigluzdov — PubChem popularity prior: '
        'https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior\n'
        '@hengck23 — inherited hengck23-dreams-enveda-casmi26 structural candidate resource: '
        'https://www.kaggle.com/hengck23\n'
        'Official metric implementation, version 14: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank\n'
        'ICEBERG/GLACIER research tooling: https://github.com/coleygroup/ms-pred\n'
        'Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra\n'
        'RDKit official metric wheel: https://www.kaggle.com/datasets/metric/rdkit-2026-3-3-wheel\n'
        'ChEBI/LIPID MAPS and COCONUT resources remain credited in inherited code and input metadata.\n'
        'The public notebook declares Apache-2.0; upstream datasets/components retain their own licences. '
        'Substantial code, including embedded modules, is reused verbatim. Our adaptation changes presentation '
        'and adds provenance, not authorship of the engines. Upstream comments are retained as provenance, '
        'not independently established component-performance claims.')
    nb['cells'].insert(0, code_cell(prose('CASMI26: Metric-Audited Safe Tail — Public Adaptation\n'
        'Faithful computational adaptation of @obstacledeveloper V1.1 (public 0.433). '
        'English narrative is displayed as triple-quoted code-cell comments. '
        'All original computational ASTs and embedded modules are preserved. '
        'No preceding 0.420 experiment is enabled.\n'
        'Use GPU T4 x2, Internet OFF and the original 17 input sources. '
        'The source simulator setup is cp313-specific: retain Python 3.13 and its RDKit 2025.3.6 cp313 wheel. '
        'The main metric environment requires RDKit 2026.03.3. '
        'Visible 12-molecule smoke is not a validation score.')))
    nb['cells'].insert(1, code_cell(credits))
    nb['metadata']['casmi_experiment'] = dict(source_url=URL, script_version_id=356393001,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), source_public_score=0.433,
        measured_score=None, validation='AST parity and local focused tests; full Kaggle run pending')
    validate(nb)
    experiment = copy.deepcopy(nb)
    experiment['cells'][0] = code_cell(prose('CASMI26: Metric-Safe Wide Mass Tail\n'
        'Independent overlay on the complete faithful @obstacledeveloper V1.1 adaptation. '
        'Only vacant slots receive additional DreaMS mass candidates at 20 ppm. '
        'All existing valid metric identities and their ranks are preserved from this run. '
        'No score gain or fresh-run reproducibility is guaranteed; full 25-candidate lists are unchanged. '
        'Same inputs and environment as the faithful adaptation.'))
    experiment['cells'].append(code_cell(WIDE_RUN))
    experiment['metadata']['casmi_experiment']['overlay'] = 'append_only_20ppm_dreams_vacancies'
    validate(experiment)
    return nb, experiment


if __name__ == '__main__':
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        shutil.copy2(DOWNLOAD, SOURCE)
    for name, nb in zip(('casmi26_0433_metric_safe_adaptation.ipynb',
                         'casmi26_0433_safe_wide_mass_tail.ipynb'), build()):
        path = ROOT / 'experiments' / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path)
