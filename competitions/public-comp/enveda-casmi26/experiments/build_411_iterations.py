"""Package the supplied 0.411 CASMI26 notebook as two self-contained Kaggle experiments."""

import copy
import json
import uuid
from pathlib import Path


SOURCE = Path('/Users/flexonafft/Downloads/enveda-casmi-2026.ipynb')
OUT = Path(__file__).resolve().parent

SECTIONS = [
    ('Runtime and inputs', 'Locate the pinned model datasets, popularity arrays, and RDKit wheel.'),
    ('Commit smoke switch', 'The visible test is a 12-molecule smoke run; the competition rerun uses the full hidden test.'),
    ('Independent engine', 'Build and run the second candidate/ranking engine in an isolated process.'),
    ('PubChem channel', 'Search PubChem candidates with fingerprint and popularity-union evidence.'),
    ('Spectrum library', 'Build the spectrum cache and link the mass-indexed candidate pool.'),
    ('Main model', 'Load the fingerprint model, feature families, and LightGBM ranker.'),
    ('Base candidate lists', 'Rank the main-engine candidates before forward-model rescoring.'),
    ('Forward-model evidence', 'Score candidate spectra with GLACIER and preserve the source ICEBERG settings.'),
    ('Gated PubChem merge', 'Insert PubChem-only structures at the reference slots.'),
    ('Engine fusion and submission', 'Fuse the two engines, validate output shape, and write submission.csv.'),
]

GL_PRESELECT = """    if GL_ONLY_PRESELECT and lib_max < LIB_TAU and gl and not any(v is not None for v in ice.values()):
        try:
            '''The source has no ICEBERG predictions (ICE_CHUNKS=0), so its early GL rerank is skipped.'''
            o_gl = gl_fuse.rerank_multi(smis, keys, scs, forms, [gl], [GL_LAM], top_n=len(smis))
            new_smis = [smis[i] for i in o_gl]
            new_keys = [keys[i] for i in o_gl]
            GL_PRE_STATS['molecules'] += 1
            GL_PRE_STATS['changed_top1'] += int(new_smis[:1] != smis[:1])
            GL_PRE_STATS['changed_top25'] += int(new_smis[:25] != smis[:25])
            smis, keys = new_smis, new_keys
        except Exception as ex:
            print('GL-only preselection failed', m, repr(ex))
"""


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one {old!r}, got {source.count(old)}')
    return source.replace(old, new, 1)


def comment_cell(text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'id': uuid.uuid5(uuid.NAMESPACE_URL, 'casmi26-411-' + text).hex[:8],
        'metadata': {},
        'outputs': [],
        'source': ("'''" + text + "'''\n").splitlines(keepends=True),
    }


def make(experimental):
    notebook = copy.deepcopy(json.loads(SOURCE.read_text()))
    notebook['metadata'].pop('papermill', None)
    notebook['metadata']['kaggle']['isGpuEnabled'] = True
    original = notebook['cells']
    if len(original) != len(SECTIONS):
        raise ValueError('The downloaded notebook cell layout changed')

    if experimental:
        cell = original[7]
        src = ''.join(cell['source'])
        src = replace_once(src, 'for m in list(BASE):',
                           "GL_ONLY_PRESELECT = True\nGL_PRE_STATS = dict(molecules=0, changed_top1=0, changed_top25=0)\nfor m in list(BASE):")
        src = replace_once(src, '    BASE[m] = (smis[:25], keys[:25], lib_max)',
                           GL_PRESELECT + '    BASE[m] = (smis[:25], keys[:25], lib_max)')
        src = replace_once(src, "print('GL rerank stats', gl_stats, f'{time.time()-T0:.0f}s')",
                           "print('GL rerank stats', gl_stats, f'{time.time()-T0:.0f}s')\nprint('GL-only preselection stats', GL_PRE_STATS)")
        cell['source'] = src.splitlines(keepends=True)

    title = ('CASMI26 | Popularity-Union + GL-Only Preselection' if experimental
             else 'CASMI26 | Popularity-Union + GLACIER Fusion')
    description = ('Experiment: apply already-computed GLACIER evidence before the main candidate list is cut to 25, '
                   'only without a strong library hit. The rest of the 0.411 source code is unchanged.' if experimental
                   else 'Faithful computational adaptation of the attached enveda-casmi-2026.ipynb. '
                   'The reported 0.411 public score has not been independently reproduced here.')
    header = comment_cell(
        f'{title}\n{description}\n\n'
        'Inputs: competition data, casmi26-v4b-models, casmi26-v3-models, casmi26-v2-pool, '
        'casmi26-pubchem-tier, casmi26-pubchem-popularity-prior, casmi26-fpnet-full1, '
        'casmi26-glacier, casmi26-iceberg, the second-engine datasets, and the RDKit wheel.\n'
        'GPU T4 required. Save Version performs a 12-molecule smoke test; the scoring rerun processes the hidden test.'
    )
    cells = [header]
    for i, (section, note) in enumerate(SECTIONS):
        cells.append(comment_cell(f'{i + 1}. {section}\n{note}'))
        original[i]['execution_count'] = None
        original[i]['outputs'] = []
        cells.append(original[i])
    notebook['cells'] = cells

    filename = ('casmi26_exp_411_gl_only_preselection.ipynb' if experimental
                else 'casmi26_exp_411_popularity_union_reproduction.ipynb')
    path = OUT / filename
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    print(path)


if __name__ == '__main__':
    make(False)
    make(True)
