'''Build two independent experiments from the user-confirmed 0.420 notebook.'''
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'archive/experiments-cleanup-2026-10-08/casmi26_candidate_aligned_confidence.ipynb'


def source(nb, index):
    return ''.join(nb['cells'][index]['source'])


def put(nb, index, value):
    nb['cells'][index]['source'] = value.splitlines(keepends=True)


def replace(value, old, new):
    assert value.count(old) == 1, (old[:100], value.count(old))
    return value.replace(old, new, 1)


def prose(value):
    assert "'''" not in value
    return "'''" + value + "'''\n"


def prepare(title, hypothesis, scope):
    nb = json.loads(BASE.read_text())
    put(nb, 0, prose(title + '\n\n' + hypothesis + '\n\n'
        'Starting point: Candidate-Aligned Confidence Fusion, user-confirmed public score 0.420. '
        'This experiment has no measured score and no guaranteed improvement. '
        'The final top-1 shield and four-seed/deeper-analog experiments are NOT enabled.'))
    credits = source(nb, 1).rsplit("'''", 1)[0]
    credits += '\n\nContribution of this iteration: ' + scope
    credits += '\nInherited source remains derivative and may include verbatim code. '
    credits += 'No claim of exclusive authorship is made.\n' + "'''\n"
    put(nb, 1, credits)
    put(nb, 2, prose('Experiment scope\n' + scope))
    put(nb, 3, prose('Preserved baseline\nCandidate-aligned evidence, shortlist=60, fusion alpha=0.6/KRR=3, '
        'main-pool popularity mu=0.15, PubChem lambda=0.25/union=200, model weights, '
        'candidate sources, and forward-model budgets are unchanged. No DreaMS is introduced. '
        'Use the same Kaggle inputs, dataset versions and GPU settings as the confirmed baseline.'))
    put(nb, 5, prose('Hypothesis, not a result\n' + hypothesis + '\n'
        'Evaluate this notebook independently against 0.420. Public leaderboard gains do not '
        'establish a private leaderboard gain. Full competition inference requires Kaggle inputs and GPU; '
        'local validation covers syntax, embedded Python and focused synthetic regression checks only.'))
    nb['metadata']['casmi_experiment'] = dict(title=title, source=BASE.name,
        source_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        user_reported_baseline_score=0.420, measured_score=None,
        validation='Local syntax and synthetic regression checks; full Kaggle run pending')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return nb


DEFER_HELPER = '''
def defer_pubchem_glacier(mid, ice, gl, engines):
    engine = engines.get(str(mid)) if engines else None
    return bool(ice and gl and engine and engine.get('smiles'))
'''

REFERENCE = '''
''' + prose('Baseline comparison\nReconstruct the original 0.420 gate and fusion using the saved early-reranked PubChem lists. '
    'No second neural-model pass is required. The inherited reference is renamed because it compares gates, not this experiment.') + '''
os.replace('submission_reference.csv', 'submission_gate_reference.csv')
def original_pre_fusion(mid):
    smis, keys, lib_max = BASE.get(mid, ([], [], 0.0))
    evidence = PC_BASELINE_POST.get(mid)
    if not evidence or not evidence.get('pc') or lib_max >= LIB_TAU:
        return list(smis)
    pool_fit, pc_fit = evidence.get('best_pool_fz'), evidence.get('fz_top')
    rel = (pc_fit - pool_fit if pool_fit is not None and pc_fit is not None
           and np.isfinite(pool_fit) and np.isfinite(pc_fit) else None)
    slots = SLOTS_AGG if rel is not None and rel > REL_TH else SLOTS_GENTLE
    return gate.apply(smis, keys, lib_max, align_candidate_evidence(evidence), slots)[0]

actual = pd.read_csv('submission.csv')
reference = actual[['molecule_id']].copy()
reference['smiles'] = [';'.join(finalize_reference(mid, original_pre_fusion(mid)))
                       for mid in reference.molecule_id]
reference.to_csv('submission_reference.csv', index=False)
audit = []
for mid, current, previous in zip(actual.molecule_id, actual.smiles, reference.smiles):
    now, old = current.split(';'), previous.split(';')
    audit.append(dict(molecule_id=str(mid), early_gl_deferred=str(mid) in PC_GL_DEFERRED,
                      changed=now != old, top1_changed=now[:1] != old[:1],
                      candidate_set_changed=set(now) != set(old)))
assert all(not r['changed'] for r in audit if not r['early_gl_deferred']), 'Untargeted rows changed'
json.dump(dict(experiment='single_pass_glacier', baseline_public_score=0.420,
               reference_scope='Original candidate-aligned route with early and final GLACIER',
               deferred_molecules=len(PC_GL_DEFERRED),
               final_lists_changed=sum(r['changed'] for r in audit),
               top1_changed=sum(r['top1_changed'] for r in audit),
               candidate_sets_changed=sum(r['candidate_set_changed'] for r in audit),
               same_as_reference=actual.smiles.equals(reference.smiles), audit=audit),
          open('experiment_diagnostics.json', 'w'), indent=2)
print('Single-pass GLACIER:', len(PC_GL_DEFERRED), 'eligible molecules;',
      sum(r['changed'] for r in audit), 'changed final lists')
'''


def single_pass():
    nb = prepare('CASMI26: Single-Pass GLACIER Fusion',
        'Test whether using GLACIER only after engine fusion avoids premature PubChem reordering.',
        'Defer early PubChem GLACIER only when valid GLACIER scores, a nonempty ICE routing dictionary '
        'and second-engine predictions make final forward reranking available. Preserve the baseline '
        'for all other molecules. ICE_CHUNKS remains zero; this is not an ICEBERG ablation.')
    put(nb, 6, prose('Diagnostics\nsubmission.csv is the experimental submission. '
        'submission_reference.csv reconstructs the original 0.420 route from the same scores. '
        'experiment_diagnostics.json counts eligibility, changed lists, top-1 and candidate-set changes. '
        'If final reranking fails on a deferred molecule, stop instead of silently submitting an unintended route.'))
    s = source(nb, 23)
    s = replace(s, 'for m in list(BASE):',
        "import copy\nPC_BASELINE_POST, PC_GL_DEFERRED = {}, set()\n" + DEFER_HELPER +
        "\nassert ICE_CHUNKS == 0, 'This experiment assumes the inherited disabled ICEBERG pass'\nfor m in list(BASE):\n    pc_before = copy.deepcopy(PC.get(m)) if isinstance(PC, dict) else None")
    s = replace(s, "print('ICE rerank stats', ice_stats, f'{time.time()-T0:.0f}s')",
        "    PC_BASELINE_POST[m] = copy.deepcopy(PC.get(m)) if isinstance(PC, dict) else None\n"
        "    if p and p.get('pc_form') and defer_pubchem_glacier(m, ice, gl, ENG):\n"
        "        assert not any(v is not None for v in ice.values()), 'Unexpected ICE scores: isolate GLACIER only'\n"
        "        PC[m] = pc_before\n"
        "        PC_GL_DEFERRED.add(str(m))\n"
        "print('Early PubChem GL deferred:', len(PC_GL_DEFERRED))\n"
        "print('ICE rerank stats', ice_stats, f'{time.time()-T0:.0f}s')")
    put(nb, 23, s)
    s = replace(source(nb, 27), "                    print('fused ICE rerank failed', mid, repr(ex))",
        "                    if str(mid) in PC_GL_DEFERRED:\n"
        "                        raise RuntimeError(f'Final GLACIER failed for deferred molecule {mid}') from ex\n"
        "                    print('fused ICE rerank failed', mid, repr(ex))")
    put(nb, 27, s + '\n' + REFERENCE)
    return nb


def adduct_safe():
    nb = prepare('CASMI26: Adduct-Safe PubChem Fusion',
        'Test whether avoiding mixed-adduct merged spectra improves the PubChem fingerprint channel. '
        'A previous broader adduct-grouping experiment did not improve the public score; '
        'this narrower test is limited to PubChem and can also regress.',
        'Change only PubChem molecule_logits: merge by (polarity, adduct), average adduct-group logits '
        'within each polarity, then average polarities as before. Keep the per-spectrum half unchanged. '
        'Molecules with at most one adduct per polarity must retain identical logits.')
    credits = source(nb, 1).rsplit("'''", 1)[0]
    put(nb, 1, credits + '\nScientific motivation, not a CASMI performance claim: '
        'Ion identity molecular networking for mass spectrometry-based metabolomics in GNPS (2021), '
        'https://pmc.ncbi.nlm.nih.gov/articles/PMC8219731/\n' + "'''\n")
    put(nb, 6, prose('Diagnostics\nMixed-adduct molecule counts are recorded in adduct_experiment.json. '
        'submission.csv is the experimental output. submission_gate_reference.csv compares gates under '
        'the NEW logits, not against the old 0.420 predictions. A full baseline CSV comparison requires '
        'the saved original submission. No extra input datasets or model weights are required.'))
    s = source(nb, 15)
    tree = ast.parse(s)
    assignment = next(x for x in tree.body if isinstance(x, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CORE' for t in x.targets))
    core = ast.literal_eval(assignment.value)
    core = replace(core, 'mean per-polarity merged-spectrum logits',
        'mean per-polarity logits after averaging separate adduct-group merged spectra')
    core = replace(core, "    items_m = []\n    for md in (1, -1):\n        grp = [s for s in spectra if s['mode'] == md]\n        if not grp:\n            continue",
        "    '''EXPERIMENT: never merge different ion species; preserve equal polarity weighting.'''\n"
        "    items_m, merged_modes = [], []\n"
        "    groups = [(md, ad) for md in (1, -1)\n"
        "              for ad in dict.fromkeys(s['adduct'] for s in spectra if s['mode'] == md)]\n"
        "    for md, ad in groups:\n"
        "        grp = [s for s in spectra if s['mode'] == md and s['adduct'] == ad]\n"
        "        merged_modes.append(md)")
    core = replace(core, '    return (0.5 * (zs.mean(0) + zm.mean(0))).astype(np.float32)',
        "    polarity_logits = [zm[np.asarray(merged_modes) == md].mean(0)\n"
        "                       for md in (1, -1) if md in merged_modes]\n"
        "    merged_logits = np.asarray(polarity_logits).mean(0)\n"
        "    return (0.5 * (zs.mean(0) + merged_logits)).astype(np.float32)")
    lines = s.splitlines(keepends=True)
    lines[assignment.lineno - 1:assignment.end_lineno] = ['CORE = ' + repr(core) + '\n']
    put(nb, 15, ''.join(lines))
    s = source(nb, 27) + '\n' + prose('Experimental coverage. No baseline-score claim is inferred from this diagnostic.') + '''
os.replace('submission_reference.csv', 'submission_gate_reference.csv')
adduct_audit = []
for mid, sub in te.groupby('molecule_id', sort=False):
    counts = sub.groupby('ionization_mode', sort=False).adduct.nunique(dropna=False)
    adduct_audit.append(dict(molecule_id=str(mid), mixed_adducts=bool((counts > 1).any()),
                            max_adducts_per_polarity=int(counts.max())))
coverage = dict(experiment='adduct_safe_pubchem', baseline_public_score=0.420,
                mixed_adduct_molecules=sum(r['mixed_adducts'] for r in adduct_audit),
                total_molecules=len(adduct_audit), audit=adduct_audit)
json.dump(coverage, open('adduct_experiment.json', 'w'), indent=2)
diagnostics = json.load(open('experiment_diagnostics.json'))
diagnostics.update(experiment='adduct_safe_pubchem', baseline_public_score=0.420,
                   reference_scope='Gate-only comparison with NEW PubChem logits; not the 0.420 baseline',
                   adduct_coverage=coverage)
json.dump(diagnostics, open('experiment_diagnostics.json', 'w'), indent=2)
print('Mixed-adduct molecules:', coverage['mixed_adduct_molecules'], '/', len(adduct_audit))
'''
    put(nb, 27, s)
    return nb


def validate(nb):
    assert nb['nbformat'] == 4
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        value = ''.join(cell['source'])
        tree = ast.parse(value)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            try:
                embedded = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(embedded, str) and any(isinstance(t, ast.Name) and t.id in ('CORE', 'RUNNER') for t in node.targets):
                ast.parse(embedded)
            if isinstance(embedded, dict):
                for name, code in embedded.items():
                    if isinstance(name, str) and name.endswith('.py') and isinstance(code, str):
                        ast.parse(code)


if __name__ == '__main__':
    for name, nb in [('casmi26_0420_single_pass_glacier.ipynb', single_pass()),
                     ('casmi26_0420_adduct_safe_pubchem.ipynb', adduct_safe())]:
        validate(nb)
        path = ROOT / 'experiments' / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path)
