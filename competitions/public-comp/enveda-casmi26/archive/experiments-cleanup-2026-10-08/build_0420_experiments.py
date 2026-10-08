"""Independent final-top1 and stronger-engine experiments from the reported 0.420 base."""
import ast
import hashlib
import json

from build_candidate_experiments import ROOT, prose, put, replace_once, text, write

BASE = ROOT / 'casmi26_candidate_aligned_confidence.ipynb'


def prepare(title, scope):
    nb = json.loads(BASE.read_text())
    put(nb, 0, prose(title + '\n\n' + scope +
        '\nStarting point: Candidate-Aligned Confidence Fusion, reported public score 0.420 by the user. '
        'This new experiment has no measured score. The other new experiment is not enabled.'))
    credits = text(nb, 1).rsplit("'''", 1)[0]
    credits += '\n\nAdditional idea provenance: the user-supplied casmi26-apex-v180-sovereign-titan.ipynb '
    credits += '(executed smoke artifact dated 2026-10-07) motivated the final top-1 shields and stronger analog engine. '
    credits += 'The uploader and exact public notebook URL are not recorded in this local artifact; '
    credits += 'add the verified author/version link before publication rather than guessing authorship. '
    credits += 'Only the stated hypothesis is tested here, not its entire architecture or claimed score.\n'
    credits += 'Contribution of this iteration: ' + scope + "'''\n"
    put(nb, 1, credits)
    put(nb, 2, prose('Experiment scope\n' + scope))
    put(nb, 3, prose('Preserved baseline\nCandidate-aligned PubChem evidence, main-pool popularity mu=0.15, '
        'PubChem lambda=0.25/union=200, fusion alpha=0.6/KRR=3, shortlist=60, and inherited GLACIER/ICEBERG '
        'settings remain unchanged. No formula diversification, adaptive fusion, adduct reconciliation, '
        'DreaMS injection or arbitrary pool padding is introduced.'))
    put(nb, 5, prose('Hypothesis and limitation\n' + scope +
        '\nNo improvement is guaranteed: agreeing engines can share a bias, a library match can be wrong, '
        'and more analogs can introduce false competitors. Compare against 0.420, not against another experiment.'))
    nb['metadata']['casmi_experiment'] = dict(source=BASE.name,
        source_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(), title=title,
        user_reported_baseline_score=0.420, measured_score=None,
        validation='local syntax and focused tests; full Kaggle GPU execution pending')
    return nb


SHIELD_HELPER = '''
def protect_final_top1(final, proposed, lib_max, second_keys):
    """Protect pre-fusion top-1 only for a strong library hit or canonical-key consensus."""
    if not proposed:
        return list(final), None
    candidate = proposed[0]
    key = chem.score_key(candidate)
    if not key:
        return list(final), None
    library = np.isfinite(lib_max) and lib_max >= LIB_TAU
    consensus = bool(second_keys) and bool(second_keys[0]) and key == second_keys[0]
    if not (library or consensus):
        return list(final), None
    reason = 'library_and_consensus' if library and consensus else 'library' if library else 'consensus'
    if final and chem.score_key(final[0]) == key:
        return list(final), reason
    return ([candidate] + [s for s in final if (chem.score_key(s) or s) != key])[:25], reason
'''


def shield():
    nb = prepare('CASMI26: Final Top-1 Evidence Shield',
        'Restore the pre-fusion top candidate after final reranking when lib_max >= 0.90 '
        'or both engines agree on its canonical score key. All other rows retain the exact baseline final list.')
    put(nb, 6, prose('Diagnostics\nsubmission_reference.csv is the unshielded 0.420 route from this same run. '
        'submission_gate_reference.csv retains the older gate comparison. experiment_diagnostics.json records '
        'qualifying shields, actually changed rows and final top-1 changes. No additional model pass is required.'))
    s = "SHIELD_PRE_FUSION = pd.read_csv('submission.csv')\n" + text(nb, 27)
    s += '\n' + prose('Final-stage protection. Applied after all inherited fusion and forward-model reranking.')
    s += SHIELD_HELPER + '''
os.replace('submission_reference.csv', 'submission_gate_reference.csv')
gate_diagnostics = json.load(open('experiment_diagnostics.json'))
unshielded = pd.read_csv('submission.csv')
unshielded.to_csv('submission_reference.csv', index=False)
proposals = dict(zip(SHIELD_PRE_FUSION.molecule_id.astype(str), SHIELD_PRE_FUSION.smiles))
library_scores = {str(mid): row[2] for mid, row in BASE.items()}
shield_rows, shield_audit = [], []
for mid, raw in zip(unshielded.molecule_id, unshielded.smiles):
    proposed = [s for s in proposals.get(str(mid), '').split(';') if s and s != 'CCO']
    original = raw.split(';')
    engine = ENG.get(str(mid), {}) if ENG else {}
    final, reason = protect_final_top1(original, proposed, library_scores.get(str(mid), 0.0),
                                      engine.get('keys') or [])
    shield_rows.append(';'.join(final) if final else 'CCO')
    shield_audit.append(dict(molecule_id=str(mid), reason=reason, changed=final != original,
                             old_top1=original[0], new_top1=final[0] if final else None))
shielded = unshielded.copy()
shielded['smiles'] = shield_rows
assert shielded.molecule_id.equals(unshielded.molecule_id)
assert shielded.smiles.notna().all() and shielded.smiles.map(lambda s: 1 <= len(s.split(';')) <= 25).all()
shielded.to_csv('submission.csv', index=False)
changed = int((shielded.smiles != unshielded.smiles).sum())
json.dump(dict(experiment='final_top1_shield', baseline_public_score=0.420,
               final_lists_changed=changed, same_as_reference=changed == 0,
               qualified=sum(r['reason'] is not None for r in shield_audit),
               audit=shield_audit, inherited_gate_diagnostics=gate_diagnostics),
          open('experiment_diagnostics.json', 'w'), indent=2)
print('Final top-1 shield changed rows:', changed, '| identical to baseline:', changed == 0)
'''
    put(nb, 27, s)
    return nb


def stronger_engine():
    nb = prepare('CASMI26: Four-Seed Deep Analog Fusion',
        'Expand the additional engine ranker from seeds (0,1) to (0,1,2,3) and analog retrieval '
        'from 200 to 250. The prvsiyan ranker already has four seeds and stays unchanged. '
        'No final top-1 shield is enabled.')
    put(nb, 6, prose('Diagnostics and execution\nTraining more rankers and scoring more analogs increases runtime. '
        'The subprocess logs its active settings and must succeed: a failed engine aborts this experiment '
        'instead of silently submitting a weaker fallback. engine_experiment.json records settings and coverage. '
        'submission_gate_reference.csv compares aligned vs inherited gate using the NEW engine; '
        'it is NOT the old 0.420 baseline, which would require a separate full engine run.'))
    s = text(nb, 13)
    s = replace_once(s, 'E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200',
        'E.RANK.SEEDS = (0, 1, 2, 3); E.CFG.N_ANALOG = 250')
    marker = "    import glob\\n    fp_models ="
    insertion = "    assert E.RANK.SEEDS == (0, 1, 2, 3) and E.CFG.N_ANALOG == 250\\n"
    insertion += '    print("EXPERIMENT ranker seeds", E.RANK.SEEDS, "analog depth", E.CFG.N_ANALOG, flush=True)\\n'
    s = replace_once(s, marker, insertion + marker)
    s = replace_once(s, "ENG = json.load(open('/kaggle/working/eng_lists.json'))",
        "rr.check_returncode()\n    ENG = json.load(open('/kaggle/working/eng_lists.json'))")
    s += '''
assert ENG, 'Four-seed engine failed or returned no predictions; do not submit a fallback as this experiment'
json.dump(dict(experiment='four_seed_deep_analog', ranker_seeds=[0, 1, 2, 3], analog_depth=250,
               engine_molecules=len(ENG), engine_seconds=round(time.time() - _t0, 2)),
          open('engine_experiment.json', 'w'), indent=2)
'''
    put(nb, 13, s)
    s = text(nb, 27) + '''
os.replace('submission_reference.csv', 'submission_gate_reference.csv')
diagnostics = json.load(open('experiment_diagnostics.json'))
diagnostics['experiment'] = 'four_seed_deep_analog'
diagnostics['baseline_public_score'] = 0.420
diagnostics['reference_scope'] = 'Gate-only comparison with NEW engine; not the original 0.420 submission'
diagnostics['engine'] = json.load(open('engine_experiment.json'))
diagnostics['final_top1_shield_enabled'] = False
json.dump(diagnostics, open('experiment_diagnostics.json', 'w'), indent=2)
'''
    put(nb, 27, s)
    return nb


if __name__ == '__main__':
    write(shield(), 'casmi26_0420_final_top1_shield.ipynb')
    write(stronger_engine(), 'casmi26_0420_four_seed_deep_analog.ipynb')
