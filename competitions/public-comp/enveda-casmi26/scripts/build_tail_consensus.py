'''Create the tail-only consensus experiment without changing upstream inference.'''
import json

from build_flow_experiments import ROOT, prepare, prose, put, source, validate


TAIL_HELPER = '''
def consensus_tail(final, primary, secondary, score_key):
    final = list(final)
    if len(final) <= 20:
        return final, []
    final_keys = [score_key(s) or s for s in final]
    assert len(set(final_keys)) == len(final_keys), 'Baseline contains canonical duplicates; inspect before submission'
    ranks, smiles = [], {}
    for candidates in (primary, secondary):
        rank_of = {}
        for rank, smi in enumerate(candidates[:25], 1):
            key = score_key(smi)
            if key and smi != 'CCO':
                rank_of.setdefault(key, rank)
                smiles.setdefault(key, smi)
        ranks.append(rank_of)
    novel = [key for key in ranks[0] if key in ranks[1] and key not in set(final_keys)]
    novel.sort(key=lambda key: -(1.0 / (KRR + ranks[0][key]) + ALPHA / (KRR + ranks[1][key])))
    added = [smiles[key] for key in novel[:len(final) - 20]]
    if not added:
        return final, []
    result = final[:20] + added + final[20:len(final) - len(added)]
    assert result[:20] == final[:20] and len(result) == len(final)
    assert len({score_key(s) or s for s in result}) == len(result)
    return result, added
'''


TAIL_RUN = '''
''' + prose('Tail-only consensus coverage\nThe primary route is the candidate-aligned list immediately before engine fusion '
    '(including its inherited PubChem gate). The secondary route is the additional analog engine. '
    'Only structures in both routes top-25 but absent from the final baseline list are eligible. '
    'Use the inherited weighted reciprocal-rank formula; preserve the first 20 strings exactly and '
    'preserve list length. Fill unused tail slots with the original tail in its original order. '
    'Short lists of at most 20 candidates and rows without eligible additions are untouched.') + TAIL_HELPER + '''
assert ENG, 'Secondary engine unavailable: this consensus experiment cannot be evaluated'
os.replace('submission_reference.csv', 'submission_gate_reference.csv')
inherited_diagnostics = json.load(open('experiment_diagnostics.json'))
baseline = pd.read_csv('submission.csv')
baseline.to_csv('submission_reference.csv', index=False)
primary_by_id = dict(zip(TAIL_PRIMARY.molecule_id.astype(str), TAIL_PRIMARY.smiles))
output, audit = [], []
for mid, raw in zip(baseline.molecule_id, baseline.smiles):
    original = raw.split(';')
    primary = [s for s in primary_by_id.get(str(mid), '').split(';') if s and s != 'CCO']
    engine = ENG.get(str(mid)) or {}
    updated, added = consensus_tail(original, primary, engine.get('smiles') or [], chem.score_key)
    output.append(';'.join(updated))
    audit.append(dict(molecule_id=str(mid), changed=updated != original, added=added,
                      removed=[s for s in original if s not in updated]))
submission = baseline.copy()
submission['smiles'] = output
assert submission.molecule_id.equals(baseline.molecule_id) and submission.molecule_id.is_unique
assert submission.smiles.notna().all()
assert submission.smiles.map(lambda s: 1 <= len(s.split(';')) <= 25).all()
assert all(a.split(';')[:20] == b.split(';')[:20] for a, b in zip(submission.smiles, baseline.smiles))
submission.to_csv('submission.csv', index=False)
changed = sum(row['changed'] for row in audit)
json.dump(dict(experiment='top20_locked_consensus_tail', baseline_public_score=0.420,
               reference_scope='Unmodified candidate-aligned route from the same run; score not independently reproduced',
               final_lists_changed=changed, candidates_added=sum(len(r['added']) for r in audit),
               top20_exactly_preserved=True, same_as_reference=changed == 0,
               audit=audit, inherited_gate_diagnostics=inherited_diagnostics),
          open('experiment_diagnostics.json', 'w'), indent=2)
print('Tail consensus changed rows:', changed)
print('DO NOT SUBMIT: identical to baseline' if changed == 0 else
      'Tail changed; inspect experiment_diagnostics.json before submitting submission.csv')
'''


def build():
    nb = prepare('CASMI26: Top-20 Locked Consensus Tail',
        'Test whether consensus candidates discarded by final fusion/reranking improve tail coverage '
        'without changing the first 20 baseline candidates. Tail replacement may still reduce the score.',
        'After the unchanged candidate-aligned 0.420 route completes, replace up to five tail positions '
        'with novel candidates appearing in both engines top-25, ordered by the inherited weighted RRF. '
        'No final top-1 shield, deeper analogs, single-pass GLACIER or adduct-safe aggregation is enabled.')
    put(nb, 6, prose('Outputs and validation\nsubmission.csv is the tail experiment; submission_reference.csv is '
        'the unchanged baseline route from this run; submission_gate_reference.csv is the older gate-only comparison. '
        'experiment_diagnostics.json records inserted and removed candidates and changed molecules. '
        'If final_lists_changed is zero, do not spend a submission on this file. '
        'The 0.420 score is historical and is not guaranteed to reproduce. No new inputs are required.'))
    put(nb, 27, "TAIL_PRIMARY = pd.read_csv('submission.csv').copy()\n" + source(nb, 27))
    nb['cells'].append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                            source=TAIL_RUN.splitlines(keepends=True)))
    validate(nb)
    return nb


if __name__ == '__main__':
    path = ROOT / 'experiments/casmi26_0420_top20_locked_consensus_tail.ipynb'
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + '\n')
    print(path)
