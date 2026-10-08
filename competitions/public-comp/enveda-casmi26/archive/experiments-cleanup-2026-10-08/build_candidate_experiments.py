"""Build two independent experiments; keep inherited inference code attributed."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'casmi26_exp_411_dual_pubchem_confidence.ipynb'


def replace_once(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


def text(nb, i):
    return ''.join(nb['cells'][i]['source'])


def put(nb, i, s):
    nb['cells'][i]['source'] = s.splitlines(keepends=True)


def prose(s):
    assert "'''" not in s
    return "'''" + s + "'''\n"


def prepare(title, explanation):
    nb = json.loads(SOURCE.read_text())
    # Preserve the two active baseline components, not either earlier ablation.
    setup = json.loads((ROOT / 'casmi26_exp_411_pool_popularity_only.ipynb').read_text())
    put(nb, 9, text(setup, 9))
    put(nb, 0, prose(title + '\n\n' + explanation +
        '\nThe user reports 0.417 for preceding iterations; this file has no measured score. '
        'DreaMS is not used. Models, weights, inputs and inherited fusion remain unchanged unless explicitly described.'))
    inherited = text(nb, 1)
    inherited = inherited[inherited.index('Inherited components and data:'):].rsplit("'''", 1)[0]
    put(nb, 1, prose('Credits & Attribution\nThis is a derivative experiment, not an original replacement for the public engines. '
        'Inherited portions are retained, including verbatim source; stylistic changes do not establish original authorship.\n\n'
        'Public ideas informing this stack:\n'
        '@bobthebot369 — dual PubChem confidence promotion and Sovereign Zenith:\n'
        'https://www.kaggle.com/code/bobthebot369/enveda-casmi-2026-v18-sovereign-zenith/notebook\n'
        '@nursrijan — public Sovereign Zenith adaptation and pool popularity idea:\n'
        'https://www.kaggle.com/code/nursrijan/enveda-casmi-2026-w-sovereign-zenith\n\n'
        + inherited.split('The distinct contribution')[0] + '\n'
        'Contribution of this version: ' + explanation + '\n'
        'Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra\n'
        'ICEBERG/GLACIER resources: https://www.kaggle.com/datasets/ahmedberatozer/casmi26-iceberg and '
        'https://www.kaggle.com/datasets/ahmedberatozer/casmi26-glacier\n'
        'Dataset source IDs and versions are preserved in the notebook metadata. Respect upstream licences.'))
    notes = [
        'Experiment scope\n' + explanation + '\nThe other experiment is NOT enabled in this file.',
        'Baseline fidelity\nMain-pool popularity mu=0.15 and the dual PubChem gate are retained. '
        'PubChem lambda=0.25, union=200, engine fusion alpha=0.6 and KRR=3 remain unchanged. '
        'The inherited ICEBERG pass uses zero chunks; GLACIER is restricted to [M+H]+.',
        'Execution\nAttach the same inputs as the starting notebook. No additional external dataset is introduced. '
        'Kaggle GPU is required for practical inference. The inherited visible-data commit uses 12 molecules; '
        'hidden competition reruns use the full test. A visible smoke score is not a validation score.',
        'Hypothesis\nThis experiment tests a mechanism, not a public-leaderboard parameter sweep. '
        'A higher score is possible but neither expected gain nor private generalization is established.',
        'Diagnostics\nThe final cell saves an unchanged-route reference submission, candidate changes, and a summary. '
        'These diagnostics compare final lists within the same run, not truth labels.',
        'Validation status\nNotebook and embedded Python syntax plus small branch tests are checked locally. '
        'Full execution requires attached Kaggle inputs and GPU; it has not been performed locally. '
        'Compare final CSVs before using a submission attempt.'
    ]
    for i, note in enumerate(notes, 2):
        put(nb, i, prose(note))
    nb['metadata']['casmi_experiment'] = dict(source=SOURCE.name,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), title=title,
        measured_score=None, validation='local syntax and focused tests only')
    return nb


ALIGN_HELPERS = '''
def align_candidate_evidence(evidence):
    """Read immutable evidence by canonical key, never by a stale list index."""
    current = dict(evidence)
    keys = current.get('pc_keys') or []
    row = (current.get('candidate_evidence') or {}).get(keys[0]) if keys else None
    current['S'] = row.get('score') if row else None
    current['top_pop'] = row.get('popularity') if row else None
    current['fz_top'] = row.get('raw_fz') if row else None
    return current
'''


def build_aligned():
    nb = prepare('CASMI26: Candidate-Aligned Confidence Fusion',
        'Bind raw fingerprint, popularity and combined evidence to canonical candidate keys. '
        'After forward-model reordering, apply the existing gate to the actual top candidate, '
        'without changing thresholds or final fusion.')
    s = text(nb, 15)
    insertion = s.index("open('/kaggle/working/probe_core2.py'")
    patch = '''
''' + prose('Candidate evidence patch: keep retrieval order and scoring unchanged; export immutable keyed evidence.') + '''
V17_PATCH = V17_PATCH.replace(
    "out, out_s, out_k, out_pop, seen = [], [], [], [], set()",
    "out, out_s, out_k, out_pop, seen = [], [], [], [], set(); candidate_evidence = {}")
_anchor = "out.append(smis[idx[j]]); out_s.append(float(sc[j])); out_k.append(k); out_pop.append(float(pop[idx[j]]))"
assert V17_PATCH.count(_anchor) == 1
V17_PATCH = V17_PATCH.replace(_anchor, _anchor +
    "\\n        candidate_evidence[k] = dict(raw_fz=float(fz[j]), score=float(sc[j]), popularity=float(pop[idx[j]]))")
V17_PATCH = V17_PATCH.replace("diag['n_listed'] = len(out)",
    "diag['n_listed'] = len(out)\\n    diag['candidate_evidence'] = candidate_evidence")
_anchor = "S=d.get('S'), top_pop=d.get('top_pop'), **extra[mid]"
assert RUNNER.count(_anchor) == 1
RUNNER = RUNNER.replace(_anchor,
    "S=d.get('S'), top_pop=d.get('top_pop'), candidate_evidence=d.get('candidate_evidence', {}), **extra[mid]")
'''
    put(nb, 15, s[:insertion] + patch + s[insertion:])
    # Slot selection is deliberately unchanged: it is a channel-wide rule.
    s = text(nb, 25)
    s = ALIGN_HELPERS + s
    s = replace_once(s, 'gate = PubChemDualConfidenceGate()',
        'gate = PubChemDualConfidenceGate()\nALIGN_AUDIT, ALIGN_REFERENCE = [], {}')
    s = replace_once(s, 'final, reason = gate.apply(smis, keys, lib_max, evidence, slots)',
        "reference, old_reason = gate.apply(smis, keys, lib_max, evidence, slots)\n"
        "        aligned = align_candidate_evidence(evidence)\n"
        "        final, reason = gate.apply(smis, keys, lib_max, aligned, slots)\n"
        "        ALIGN_AUDIT.append(dict(molecule_id=str(mid), old_reason=old_reason, new_reason=reason,\n"
        "                                changed=final != reference, top_key=(evidence.get('pc_keys') or [None])[0]))\n"
        "        ALIGN_REFERENCE[str(mid)] = reference")
    s = replace_once(s, 'if not final:\n        final',
        "ALIGN_REFERENCE.setdefault(str(mid), list(final))\n    if not final:\n        final")
    put(nb, 25, s)
    s = text(nb, 27)
    # Re-run only the cheap deterministic final fusion for a true within-run reference.
    s += '''
def finalize_reference(mid, vs):
    vs = [s for s in vs if s and s != 'CCO']
    e = ENG.get(str(mid)) if ENG else None
    if not e or not e.get('smiles'):
        return vs[:25] or ['CCO']
    fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])
    ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
    if ice:
        try:
            forms = [_form2(x) for x in fsm]
            order = ice_fuse.rerank(fsm, fk, fsc, forms, ice, lam=ICE_LAM, top_n=len(fsm))
            gl = gl_of(mid)
            if gl:
                order = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [ice, gl], [ICE_LAM, GL_LAM], top_n=len(fsm))
            fsm = [fsm[i] for i in order]
        except Exception:
            pass
    return fsm[:25] or ['CCO']

actual = pd.read_csv('submission.csv')
reference = actual[['molecule_id']].copy()
reference['smiles'] = [';'.join(finalize_reference(m, ALIGN_REFERENCE.get(str(m), ['CCO'])))
                       for m in reference.molecule_id]
reference.to_csv('submission_reference.csv', index=False)
reference_by_id = dict(zip(reference.molecule_id.astype(str), reference.smiles))
changed = sum(s != reference_by_id[str(m)] for m, s in zip(actual.molecule_id, actual.smiles))
json.dump(dict(experiment='candidate_aligned_gate', final_lists_changed=int(changed),
               same_as_reference=changed == 0, gate_audit=ALIGN_AUDIT),
          open('experiment_diagnostics.json', 'w'), indent=2)
print('Candidate-aligned final lists changed:', changed, '| identical submission:', changed == 0)
'''
    put(nb, 27, s)
    return nb


EXPANSION_HELPERS = '''
EXP_TOPN, EXP_MAX_MOLECULES, EXP_GL_BUDGET = 100, 20, 900
EXP_FULL, EXP_SELECTED = {}, []

def expansion_target(main_keys, library_score, second_keys, has_positive_spectrum):
    return (bool(main_keys) and bool(second_keys) and bool(has_positive_spectrum)
            and np.isfinite(library_score) and library_score < LIB_TAU
            and main_keys[0] != second_keys[0])

def complete_forward_scores(smiles, scores):
    return bool(smiles) and all(s in scores and scores[s] is not None
                               and np.isfinite(scores[s]) for s in smiles)
'''


def build_expansion():
    nb = prepare('CASMI26: Targeted Candidate Expansion & Reranking',
        'Keep the baseline intact and collect up to 100 main-engine candidates. '
        'For at most 20 low-library-confidence molecules whose engines disagree on top-1, '
        'rescore a wider union with an additional GLACIER pass (900-second total budget). '
        'Apply only complete, finite forward-model results; otherwise preserve the baseline CSV row. '
        'The candidate-aligned gate fix is NOT enabled here.')
    s = text(nb, 21)
    s = EXPANSION_HELPERS + s
    s = replace_once(s, 'if len(smis) >= TOPN: break', 'if len(smis) >= EXP_TOPN: break')
    s = replace_once(s, 'if POOLPOP_MU > 0 and len(scs) > 1:',
        "extra_smis, extra_keys = smis[TOPN:], keys[TOPN:]\n"
        "    smis, keys, scs, forms, pids_ = (smis[:TOPN], keys[:TOPN], scs[:TOPN], forms[:TOPN], pids_[:TOPN])\n"
        "    if POOLPOP_MU > 0 and len(scs) > 1:")
    s = replace_once(s, 'BASE[mid] = (smis, keys, lib_max, scs, forms)',
        "BASE[mid] = (smis, keys, lib_max, scs, forms)\n"
        "    second = ENG.get(str(mid), {}) if ENG else {}\n"
        "    has_positive = bool((sub.adduct == '[M+H]+').any())\n"
        "    if extra_smis and expansion_target(keys, lib_max, second.get('keys'), has_positive):\n"
        "        EXP_FULL[str(mid)] = dict(smiles=smis + extra_smis, keys=keys + extra_keys, lib_max=lib_max)")
    s += "\nEXP_SELECTED = sorted(EXP_FULL, key=lambda m: (EXP_FULL[m]['lib_max'], m))[:EXP_MAX_MOLECULES]\nprint('Expansion eligible:', len(EXP_FULL), '| selected:', len(EXP_SELECTED))\n"
    put(nb, 21, s)
    # Capture the pre-fusion list; otherwise original top-25 cuts defeat expansion.
    s = text(nb, 27)
    s = "EXP_PRE_FUSION = pd.read_csv('submission.csv')\n" + s
    s += '''
''' + prose('Additional targeted GLACIER pass. The standard final submission has already been written; failures keep it unchanged.') + '''
baseline = pd.read_csv('submission.csv')
baseline.to_csv('submission_reference.csv', index=False)
pre_by_id = dict(zip(EXP_PRE_FUSION.molecule_id.astype(str), EXP_PRE_FUSION.smiles))
EXP_UNION, EXP_REPORT = {}, []
for mid in EXP_SELECTED:
    main = [s for s in pre_by_id.get(mid, '').split(';') if s and s != 'CCO']
    seen = {chem.score_key(s) or s for s in main}
    for s, key in zip(EXP_FULL[mid]['smiles'], EXP_FULL[mid]['keys']):
        if key not in seen:
            main.append(s); seen.add(key)
    second = ENG[mid]
    EXP_UNION[mid] = fuse2(main, second['smiles'], second['keys'], n=EXP_TOPN + 65)

replacements = {}
try:
    if EXP_UNION:
        candidate_map = {mid: row[0] for mid, row in EXP_UNION.items()}
        positive = te[te.adduct == '[M+H]+'].copy()
        positive['molecule_id'] = positive.molecule_id.astype(str)
        extra_items = ice_fuse.build_ice_input(positive, candidate_map)
        extra_items = [dict(it, spectra=[s for s in it['spectra'] if s.get('adduct') == '[M+H]+'])
                       for it in extra_items]
        extra_items = [it for it in extra_items if it['spectra']]
        gc.collect(); torch.cuda.empty_cache()
        extra_scores = gl_fuse.run_gl(GL_PKG, extra_items,
            workdir='/kaggle/working/gl_expansion', device='cuda', budget_s=EXP_GL_BUDGET,
            site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'))
        extra_scores = extra_scores if isinstance(extra_scores, dict) else {}
        for mid, (smiles, keys, scores) in EXP_UNION.items():
            forward = extra_scores.get(str(mid), {})
            if not isinstance(forward, dict) or not complete_forward_scores(smiles, forward):
                EXP_REPORT.append(dict(molecule_id=mid, status='baseline_incomplete_forward_scores'))
                continue
            try:
                ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
                order = gl_fuse.rerank_multi(smiles, keys, scores, [_form2(s) for s in smiles],
                                            [ice, forward], [ICE_LAM, GL_LAM], top_n=len(smiles))
                assert sorted(order) == list(range(len(smiles))), 'invalid reranking permutation'
                final = [smiles[i] for i in order[:25]]
                replacements[mid] = ';'.join(final)
                original_keys = {chem.score_key(s) or s for s in
                                 baseline.loc[baseline.molecule_id.astype(str) == mid, 'smiles'].iloc[0].split(';')}
                EXP_REPORT.append(dict(molecule_id=mid, status='expanded', candidates=len(smiles),
                    new_in_top25=sum((chem.score_key(s) or s) not in original_keys for s in final)))
            except Exception as error:
                EXP_REPORT.append(dict(molecule_id=mid, status='baseline_rerank_failed', error=repr(error)))
except Exception as error:
    print('Expansion unavailable; original final submission retained:', repr(error))
    EXP_REPORT.append(dict(status='baseline_forward_pass_failed', error=repr(error)))

final = baseline.copy()
final['smiles'] = [replacements.get(str(m), s) for m, s in zip(final.molecule_id, final.smiles)]
assert len(final) == len(baseline) and final.molecule_id.is_unique and final.smiles.notna().all()
assert final.smiles.map(lambda s: 1 <= len(s.split(';')) <= 25).all()
assert all(a == b for m, a, b in zip(final.molecule_id, final.smiles, baseline.smiles)
           if str(m) not in replacements)
changed = int((final.smiles != baseline.smiles).sum())
final.to_csv('submission.csv', index=False)
json.dump(dict(experiment='targeted_expansion', eligible=len(EXP_FULL), selected=len(EXP_SELECTED),
               final_lists_changed=changed, same_as_reference=changed == 0, molecules=EXP_REPORT),
          open('experiment_diagnostics.json', 'w'), indent=2)
print('Targeted expansion final lists changed:', changed, '| identical submission:', changed == 0)
'''
    put(nb, 27, s)
    return nb


def write(nb, name):
    for cell in nb['cells']:
        assert cell['cell_type'] == 'code', 'all narrative must remain code-cell comments'
        cell['outputs'] = []
        cell['execution_count'] = None
        ast.parse(''.join(cell['source']))
    path = ROOT / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    print(path)


if __name__ == '__main__':
    write(build_aligned(), 'casmi26_candidate_aligned_confidence.ipynb')
    write(build_expansion(), 'casmi26_targeted_candidate_expansion.ipynb')
