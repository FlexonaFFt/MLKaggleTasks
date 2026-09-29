"""Build a faithful v4m notebook and one FP-bank ensemble experiment."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import urllib.request

from build_v4h_experiments import PREFLIGHT, code_cell, comment_style, computational_ast, source_text

HERE = Path(__file__).resolve().parent
VERSION = 353845929
SOURCE_PAGE = f'https://www.kaggle.com/code/ahmedberatozer/casmi26-v4m-inference?scriptVersionId={VERSION}'
SOURCE = HERE / 'sources' / f'casmi26_v4m_{VERSION}.ipynb'
PREFLIGHT = PREFLIGHT.replace("    'v4b engine':", "    'full1 FPNet': 'casmi26-fpnet-full1/**/fpnet_full1.pt',\n    'v4b engine':")
PREFLIGHT = PREFLIGHT.replace('Attach the seven datasets listed in the first cell.',
                              'Attach the eight datasets listed in the first cell.')

AUDIT = """'''CASMI26 | Submission audit
The source permits fallback if a scoring channel fails. Report any degraded run
explicitly; do not treat its score as a complete v4m comparison.
'''
_audit = dict(source_version=353845929, experiment=EXPERIMENT,
              fingerprint_bank=FP_BANK, molecules=len(mols), pubchem_molecules=len(PC),
              engine_errors=n_err, iceberg_molecules=len(ICE_SCORES),
              glacier_molecules=len(GL_SCORES), ice_stats=ice_stats, gl_stats=gl_stats,
              source_sha256=SOURCE_SHA256)
_audit['status'] = 'complete' if (n_err == 0 and len(PC) == len(mols)
                                  and ICE_SCORES and GL_SCORES) else 'degraded'
with open('validation_report.json', 'w') as _f:
    json.dump(_audit, _f, indent=2)
print(json.dumps(_audit, indent=2))
if _audit['status'] == 'degraded':
    print('DEGRADED RUN: inspect channel logs before submitting this experiment.')
"""


def build():
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f'https://www.kaggle.com/kernels/scriptcontent/{VERSION}/download', timeout=30) as response:
            SOURCE.write_bytes(response.read())
    raw = SOURCE.read_bytes()
    original = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert len(original['cells']) == 7
    first = source_text(original['cells'][3])
    needle = "bank = fpnet.ModelBank([_FULL[0] if _FULL else os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)"
    assert first.count(needle) == 1
    intro = f"""Source: {SOURCE_PAGE}
Credit: Ahmed Berat Özer. Source snapshot SHA256: {digest}
This notebook has not been GPU-run or scored locally.

Required inputs (attach all; source dataset-version IDs are retained):
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v3-models
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v2-pool
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-pubchem-tier
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-iceberg
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-glacier
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-fpnet-full1
https://www.kaggle.com/datasets/metric/rdkit-2026-3-3-wheel
Competition: enveda-CASMI26-molecule-id-mass-spectra. GPU required.
Output: submission.csv and validation_report.json.
"""
    variants = [
        ('i_v4m_reference', 'CASMI26 | v4m Faithful Reference', 'full1',
         'Scoring logic matches the pinned public v4m source.'),
        ('j_v4m_fp_ensemble', 'CASMI26 | v4m Dual-FP Ensemble', 'full1+A',
         'Only change: average full1 and original FPNet A in the engine bank.'),
    ]
    for name, title, bank, description in variants:
        nb = copy.deepcopy(original)
        nb['cells'] = [code_cell("'''" + title + '\n' + description + '\n\n' + intro + "'''\n"
                                 + f'EXPERIMENT = {name!r}\nFP_BANK = {bank!r}\nSOURCE_SHA256 = {digest!r}\n'),
                       code_cell(PREFLIGHT)]
        for index, cell in enumerate(original['cells']):
            text = source_text(cell)
            expected = text
            if index == 3 and bank == 'full1+A':
                expected = text.replace(needle,
                    "bank = fpnet.ModelBank([_FULL[0], os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)")
            formatted = comment_style(expected)
            assert computational_ast(formatted) == computational_ast(expected), index
            if bank == 'full1':
                assert computational_ast(formatted) == computational_ast(text), index
            nb['cells'].append(code_cell(formatted))
        nb['cells'].append(code_cell(AUDIT))
        nb['metadata'].pop('papermill', None)
        nb['metadata']['kaggle']['isGpuEnabled'] = True
        nb['metadata']['kaggle']['isInternetEnabled'] = False
        nb['metadata']['casmi_adaptation'] = dict(source_version=VERSION,
            source_url=SOURCE_PAGE, source_sha256=digest, experiment=name, fingerprint_bank=bank)
        for cell in nb['cells']:
            ast.parse(source_text(cell))
        path = HERE / ('casmi26_exp_' + name + '.ipynb')
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path.name, 'OK: syntax + scoring-code parity; bank =', bank)


if __name__ == '__main__':
    build()
