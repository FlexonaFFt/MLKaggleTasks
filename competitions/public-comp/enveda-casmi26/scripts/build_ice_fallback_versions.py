'''Preserve uploaded E006 arithmetic and test a single-factor 80-candidate window.'''
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path
from build_metric_safe_adaptation import code_cell, comments, computational_ast
from build_flow_experiments import ROOT, prose, validate

UPLOAD = Path('/Users/flexonafft/Downloads/casmi26-e006-window-40-ice-fallback-true.ipynb')
SOURCE = ROOT / 'archive/public-sources-2026-10-11/casmi26_e006_uploaded.ipynb'
POLICY_URL = 'https://www.kaggle.com/code/samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438'


def build():
    original = json.loads(SOURCE.read_text())
    nb = copy.deepcopy(original)
    for cell in nb['cells']:
        value = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            cell.update(code_cell(prose(value)))
        else:
            converted = comments(value)
            assert computational_ast(converted) == computational_ast(value)
            cell.update(code_cell(converted))
        cell['metadata'] = {}
    title = prose('CASMI26: Metric Fusion with ICEBERG Fallback — E006 Adaptation\n'
        'Faithful computational adaptation of the uploaded E006 notebook, reported public MRR 0.440 '
        'by the user. Our new run has not measured that score. The inherited 0.436 metric-identity '
        'fusion remains intact. Window=40; use ICEBERG same-formula reranking only when gl_of(mid) '
        'returns no GLACIER evidence. Preserve the exact public truthy-dictionary gates. '
        'No consensus-backfill or global cross-formula GLACIER is enabled. '
        'Use the same 17 input sources, Python 3.12 and T4 GPU. The inherited observers deliberately '
        'require that environment. No additional dataset is introduced. Visible smoke is not validation.')
    credits = prose('Additional provenance for this adaptation\n'
        'Direct source: user-uploaded casmi26-e006-window-40-ice-fallback-true.ipynb; SHA256: '
        + hashlib.sha256(SOURCE.read_bytes()).hexdigest() + '. '
        'The uploaded file does not identify its E006 author by a verifiable profile URL; '
        'retain this source attribution and add the original author/link before public publication. '
        '@flexonafft — inherited metric-identity fusion v80: '
        'https://www.kaggle.com/code/flexonafft/casmi26-public-baseline-adaptation-experiments?scriptVersionId=356740494\n'
        '@samanyu1808 — ICEBERG fallback where GLACIER is absent; source cited by the uploaded E006: '
        + POLICY_URL + '\n'
        'All further upstream authors, component links and licences are preserved in Credits & Attribution '
        'below. This is derivative work; substantial executable code is reused verbatim. '
        'Our contribution in the faithful version is presentation and provenance, not the fallback invention.')
    nb['cells'] = [code_cell(title), code_cell(credits)] + nb['cells']
    nb['metadata']['casmi_experiment'] = dict(source_file=SOURCE.name,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        user_reported_source_public_score=0.440, measured_score=None,
        overlay='E006_exact_window40_ice_fallback',
        validation='Executable AST parity and focused tests; full Kaggle inference pending')
    validate(nb)
    experiment = copy.deepcopy(nb)
    experiment['cells'][0] = code_cell(prose('CASMI26: ICEBERG Fallback — Extended Fusion Window\n'
        'Run the exact E006 window=40 route first, then rebuild the same fusion with window=80 '
        'and the same ICEBERG fallback. This is our single-factor experiment, not a reproduced score. '
        'The two source lists provide at most 65 unique candidates, so 80 is an upper bound, not '
        '80 newly generated structures. Candidates previously below rank 40 can participate in '
        'same-formula simulator reranking before final top-25 truncation. '
        'Weights, canonicalization, candidate sources, neural inference and simulator scores are unchanged. '
        'No extra simulator pass is added. Sparse score coverage may limit the experiment; '
        'changing the group used for normalization can also demote correct candidates. '
        'Same inputs, Python 3.12 and T4 GPU as E006. No score gain is guaranteed.'))
    overlay = ''.join(original['cells'][-1]['source'])
    assert "'experiment': 'E006', 'window': 40, 'ice_fallback': True" in overlay
    overlay = overlay.replace("'experiment': 'E006', 'window': 40, 'ice_fallback': True",
                              "'experiment': 'E007_WINDOW80', 'window': 80, 'ice_fallback': True", 1)
    experiment['cells'].append(code_cell(prose('Window expansion: same-run paired comparison\n'
        'Preserve the completed E006 CSV and audit first. Restore the verified E004 control bytes '
        'only as the input required by the inherited reconstruction helper. The new output uses '
        'the same ICE fallback as E006, with the window expanded to 80. Compare final output against '
        'submission_e006_control.csv, not merely against E004. Inactive placeholders must stay identical.')
        + "import shutil\n"
        + "shutil.copy2('submission.csv', 'submission_e006_control.csv')\n"
        + "shutil.copy2('experiment_audit_factorial.csv', 'e006_audit.csv')\n"
        + "shutil.copy2('experiment_verification.json', 'e006_verification.json')\n"
        + "shutil.copy2('submission_e004_control.csv', 'submission.csv')\n"
        + overlay + '''
_e006 = pd.read_csv('submission_e006_control.csv', dtype=str, keep_default_na=False)
_expanded = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
assert list(_expanded.molecule_id) == list(_e006.molecule_id)
_comparison = pd.DataFrame(dict(molecule_id=_e006.molecule_id,
    changed=_e006.smiles != _expanded.smiles,
    changed_top1=_e006.smiles.str.split(';').str[0] != _expanded.smiles.str.split(';').str[0]))
_comparison.to_csv('window80_vs_e006_audit.csv', index=False)
_report = dict(experiment='window80_vs_e006', changed_rows=int(_comparison.changed.sum()),
    changed_top1=int(_comparison.changed_top1.sum()),
    control_sha256=sha256('submission_e006_control.csv'), submission_sha256=sha256('submission.csv'),
    measured_score=None)
Path('window80_vs_e006_audit.json').write_text(json.dumps(_report, indent=2))
print(_report)
if not _report['changed_rows']:
    print('NO-OP versus E006: do not spend a separate submission on identical predictions.')
'''))
    experiment['metadata']['casmi_experiment']['overlay'] = 'window80_on_E006_ice_fallback'
    validate(experiment)
    return nb, experiment


if __name__ == '__main__':
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        shutil.copy2(UPLOAD, SOURCE)
    for name, nb in zip(('casmi26_e006_ice_fallback_adaptation.ipynb',
                         'casmi26_e007_extended_fusion_window.ipynb'), build()):
        path = ROOT / 'experiments' / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path)
