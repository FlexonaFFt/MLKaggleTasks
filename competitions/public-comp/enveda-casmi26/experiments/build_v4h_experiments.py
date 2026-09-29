"""Build two notebooks from the exact public v4h version; verify scoring-code parity."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import tokenize
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCE_URL = 'https://www.kaggle.com/kernels/scriptcontent/353548586/download'
SOURCE_PAGE = 'https://www.kaggle.com/code/ahmedberatozer/casmi26-v4h-inference?scriptVersionId=353548586'
SOURCE = HERE / 'sources' / 'casmi26_v4h_353548586.ipynb'

PREFLIGHT = r"""'''CASMI26 | Check all inputs before inference
Collect every missing file in one message. Require a Kaggle CUDA GPU.
The original inference cells and their runtime fallback behaviour follow below.
'''
import glob, os, sys
import torch
_required = {
    'v4b engine': 'casmi26-v4b-models/**/casmi/engine.py',
    'v4b manifest': 'casmi26-v4b-models/**/MANIFEST.json',
    'v4b fingerprint model': 'casmi26-v4b-models/**/fpnet_0.pt',
    'v4b feature engine': 'casmi26-v4b-models/**/v1engine.py',
    'v3 engine': 'casmi26-v3-models/**/casmi/engine.py',
    'v3 ranker': 'casmi26-v3-models/**/ranker_0.pkl',
    'v3 fingerprint A': 'casmi26-v3-models/**/fpnet_0.pt',
    'v3 fingerprint B': 'casmi26-v3-models/**/fpnet_1.pt',
    'ICEBERG runner': 'casmi26-iceberg/**/ice_runner.py',
    'ICEBERG fusion': 'casmi26-iceberg/**/fuse.py',
    'GLACIER runner': 'casmi26-glacier/**/gl_runner.py',
    'GLACIER fusion': 'casmi26-glacier/**/gl_fuse.py',
    'competition train': 'train.parquet',
    'competition test': 'test.parquet',
    'submission template': 'sample_submission.csv',
}
for _file in ('pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy',
              'pool_frag_mass.npy', 'fp_bits.npy', 'train_structs.parquet', 'train_fp_sel.npy'):
    _required['pool: ' + _file] = 'casmi26-v2-pool/**/' + _file
for _file in ('pc_smiles.npy', 'pc_mass.npy', 'pc_off.npy'):
    _required['PubChem: ' + _file] = 'casmi26-pubchem-tier/**/' + _file
_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
_required['offline RDKit wheel'] = f'rdkit-*{_tag}*.whl'
_missing = []
for _label, _pattern in _required.items():
    _hits = sorted(glob.glob('/kaggle/input/**/' + _pattern, recursive=True))
    if not _hits:
        _missing.append(_label + ': ' + _pattern)
    else:
        print(_label + ': ' + _hits[0])
if _missing:
    raise FileNotFoundError('Missing required inputs:\n  ' + '\n  '.join(_missing)
                            + '\nAttach the seven datasets listed in the first cell.')
if not torch.cuda.is_available():
    raise RuntimeError('Enable Kaggle Settings > Accelerator > GPU T4 x2 (or P100), then restart.')
print('Inputs ready; GPU:', torch.cuda.get_device_name(0))
"""

AUDIT = """'''CASMI26 | Submission audit
Record whether every scoring channel ran. Degraded runs retain the source fallback
behaviour but are labelled explicitly, so they cannot be mistaken for full v4h.
'''
_audit = dict(source_version=353548586, experiment=EXPERIMENT,
              glacier_weight=GL_LAM, iceberg_weight=ICE_LAM,
              molecules=len(mols), pubchem_molecules=len(PC),
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

def source_text(cell):
    value = cell['source']
    return ''.join(value) if isinstance(value, list) else value

def comment_style(text):
    """Convert actual Python comments, leaving embedded source strings untouched."""
    lines = text.splitlines(keepends=True)
    comments = [t for t in tokenize.generate_tokens(io.StringIO(text).readline)
                if t.type == tokenize.COMMENT]
    for token in reversed(comments):
        index, col = token.start[0] - 1, token.start[1]
        line = lines[index].rstrip('\n')
        prefix = line[:col]
        indent = line[:len(line) - len(line.lstrip())]
        message = token.string[1:].strip().replace("'''", "\\'\\'\\'")
        if prefix.strip():
            indent += '    ' if prefix.rstrip().endswith(':') else ''
            lines[index] = prefix.rstrip() + '\n' + indent + "'''" + message + "'''\n"
        else:
            lines[index] = indent + "'''" + message + "'''\n"
    return ''.join(lines)

def code_cell(text):
    return dict(cell_type='code', id=hashlib.sha256(text.encode()).hexdigest()[:12],
                execution_count=None, metadata={}, outputs=[],
                source=text.splitlines(keepends=True))

def computational_ast(text):
    tree = ast.parse(text)
    # Triple-quoted comment expressions have no effect on the inference algorithm.
    class StripComments(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)
    return ast.dump(StripComments().visit(tree), include_attributes=False)

def build():
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            SOURCE.write_bytes(response.read())
    raw = SOURCE.read_bytes()
    original = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert len(original['cells']) == 7
    intro = """Source: {source}
Credit: Ahmed Berat Özer. This adaptation has not been scored or GPU-run locally.
Source snapshot SHA256: {digest}

Required inputs (attach all; source dataset-version IDs are retained in metadata):
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v3-models
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v2-pool
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-pubchem-tier
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-iceberg
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-glacier
https://www.kaggle.com/datasets/metric/rdkit-2026-3-3-wheel
Competition: enveda-CASMI26-molecule-id-mass-spectra
GPU: T4 x2 or P100. Internet: off. Source runtime: about 66 minutes on its example test.
Output: submission.csv, run_manifest.json, validation_report.json.
""".format(source=SOURCE_PAGE, digest=digest)
    variants = [
        ('g_v4h_reference', 'CASMI26 | v4h Faithful Reference', 0.5,
         'Original candidate generation, models, ICEBERG, GLACIER and PubChem gates.'),
        ('h_v4h_glacier_quarter', 'CASMI26 | v4h Gentle GLACIER', 0.25,
         'Single experiment: GLACIER weight 0.5 -> 0.25. Everything else matches the reference.'),
    ]
    for name, title, weight, description in variants:
        nb = copy.deepcopy(original)
        nb['cells'] = [code_cell("'''" + title + '\n' + description + '\n\n' + intro + "'''\n"
                                  + f'EXPERIMENT = {name!r}\nSOURCE_SHA256 = {digest!r}\n'),
                       code_cell(PREFLIGHT)]
        for index, cell in enumerate(original['cells']):
            text = source_text(cell)
            expected = text
            if index == 5 and weight != 0.5:
                needle = 'changed_vs_ice_top1=0), 0.5, 2400'
                assert text.count(needle) == 1
                expected = text.replace(needle, f'changed_vs_ice_top1=0), {weight}, 2400')
            formatted = comment_style(expected)
            assert computational_ast(formatted) == computational_ast(expected), index
            nb['cells'].append(code_cell(formatted))
        nb['cells'].append(code_cell(AUDIT))
        nb['metadata'].pop('papermill', None)
        nb['metadata']['kaggle']['isGpuEnabled'] = True
        nb['metadata']['kaggle']['isInternetEnabled'] = False
        nb['metadata']['casmi_adaptation'] = dict(source_version=353548586,
            source_url=SOURCE_PAGE, source_sha256=digest, experiment=name, glacier_weight=weight)
        for cell in nb['cells']:
            ast.parse(source_text(cell))
        path = HERE / ('casmi26_exp_' + name + '.ipynb')
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path.name, 'OK: syntax + inference AST parity; GL_LAM =', weight)

if __name__ == '__main__':
    build()
