'''Median contrast experiment, independently launched after the confirmed Initial01 control.'''
import json
from build_discriminative_evidence import build as discriminative_build, DISCRIMINATIVE
from build_flow_experiments import ROOT, prose, validate
from build_metric_safe_adaptation import code_cell

REPLICATED = DISCRIMINATIVE.replace('DiscriminativeFragmentFallback','ReplicatedFragmentFallback')
REPLICATED = REPLICATED.replace('            scores = []\n', '            scores, seen = [], set()\n')
REPLICATED = REPLICATED.replace('                iw,q,ns,npk = prep\n',
    '                iw,q,ns,npk = prep\n'
    '                signature = (iw.tobytes(), q.tobytes(), ns, npk)\n'
    '                if signature in seen:\n'
    '                    continue\n'
    '                seen.add(signature)\n')
REPLICATED = REPLICATED.replace('            if not scores:\n','            if len(scores) < 3:\n')
REPLICATED = REPLICATED.replace('np.mean(scores,axis=0)','np.median(scores,axis=0)')
REPLICATED = REPLICATED.replace("'discriminative_fragment_fallback'","'replicated_fragment_fallback'")
REPLICATED = REPLICATED.replace('contrast_lambda=1.25,','contrast_lambda=1.25,min_distinct_informative_spectra=3,aggregation="median",')

METHOD = ('Iteration 07 — Replicated Fragment Evidence\n'
    'An independent treatment on Initial01 (user-confirmed public 0.441), not an overlay on submitted '
    'Iteration06 predictions. Hypothesis: a single unusually favourable spectrum can dominate mean '
    'contrast evidence. Replace its aggregation with a per-candidate median and require at least three '
    'distinct informative spectra. Exact duplicates of the prepared normalized intensities and neutral '
    'mass queries count once; distinct spectra are not guaranteed statistically independent. '
    'Each informative spectrum must have at least two discriminative peaks carrying 5% intensity. '
    'Use the same two-cleavage kernel, 2..8 candidate groups, 40-bond cap, no-forward-evidence molecule '
    'gate, 900s hidden / 120s smoke budget and aggressive lambda 1.25 as Iteration06. '
    'The method tests persistent candidate support, not exact repeated fragment identities. '
    'Groups with fewer than three informative spectra retain their original 0.441 order. '
    'Risk: collision energies can legitimately favour different structures; median filtering can '
    'discard useful evidence. No score improvement is claimed. All upstream credits and links remain. '
    'No new datasets: retain every input of the successful baseline, including the compatible '
    'RDKit2025 simulator wheel and separate RDKit2026 main-engine wheel. '
    'Check iteration_audit.json for actual changed rows; identical outputs must not consume a submission.')


def build():
    nb = discriminative_build()[1]
    nb['cells'][0] = code_cell(prose('CASMI26: Closing the Evidence Gap\n'+METHOD))
    final = ''.join(nb['cells'][-1]['source'])
    assert final.count(DISCRIMINATIVE) == 1
    final = final.replace(DISCRIMINATIVE,REPLICATED).replace(
        'iteration = DiscriminativeFragmentFallback(experiment)',
        'iteration = ReplicatedFragmentFallback(experiment)')
    # Replace mean-based narrative instead of leaving contradictory inherited explanations.
    final = final[final.index('class EvidenceIteration:'):]
    nb['cells'][-1] = code_cell(prose(METHOD)+final)
    nb['metadata']['casmi_experiment'].update(iteration='ReplicatedFragmentFallback',
        validation='Synthetic aggregation/duplicate/rollback tests; Kaggle inference pending')
    validate(nb)
    return nb


if __name__ == '__main__':
    path = ROOT/'research/evidence-routing/casmi26_iteration07_replicated_fragments.ipynb'
    path.write_text(json.dumps(build(),ensure_ascii=False,indent=1)+'\n')
    print(path)
