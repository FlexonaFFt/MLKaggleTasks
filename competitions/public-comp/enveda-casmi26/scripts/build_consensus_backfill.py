'''Restore tail consensus on the user-confirmed 0.436 metric-fusion version.'''
import json
from build_aggressive_fusion_experiments import build as winning_build, METRIC_RUN, METRIC_HELPER
from build_metric_safe_adaptation import code_cell
from build_flow_experiments import ROOT, prose, validate

HELPER = '''
def consensus_backfill(first, second, tail, identity, alpha=0.6, offset=3.0, limit=40):
    candidates, scores, _ = metric_fusion(first, second, identity, alpha, offset,
                                          limit=len(first) + len(second))
    score_by_key = {identity(s): score for s, score in zip(candidates, scores)}
    first_keys = {identity(s) for s in first}
    second_keys = {identity(s) for s in second}
    added = set()
    for rank, candidate in enumerate(tail, 1):
        key = identity(candidate)
        if rank <= 25 or key is None or key in first_keys or key not in second_keys or key in added:
            continue
        added.add(key)
        score_by_key[key] += 1.0 / (offset + rank)
    order = sorted(range(len(candidates)), key=lambda i: -score_by_key[identity(candidates[i])])[:limit]
    return [candidates[i] for i in order], [score_by_key[identity(candidates[i])] for i in order], len(added)
'''


def build():
    nb = winning_build()[0]
    nb['cells'][0] = code_cell(prose('CASMI26: Metric-Fusion Consensus Backfill\n'
        'Starting point: our metric-identity fusion submission, confirmed public MRR 0.436 by the user. '
        'Execute that complete route first and save submission_reference.csv before this overlay. '
        'For weak-library molecules only, restore the first-engine vote for candidates ranked 26–60 '
        'in its saved base list when the second engine also retrieved the same official metric identity. '
        'The added vote is 1/(KRR + original tail rank); duplicates receive one vote only. '
        'Existing first-list ranks, representatives, fusion weights, PubChem gate and same-formula '
        'GLACIER reranking remain unchanged. No cross-formula GLACIER or additional neural inference. '
        'This can change top-1 and can reduce MRR. No score is claimed for the new experiment. '
        'Same inputs and GPU environment as the 0.436 version.'))
    run = METRIC_RUN.replace(METRIC_HELPER, HELPER, 1)
    run = run.replace("EXPERIMENT = 'metric_identity_fusion'", "EXPERIMENT = 'metric_fusion_consensus_backfill'\n_base_by_id = {str(mid): value for mid, value in BASE.items()}")
    run = run.replace("if _mid in PREFUSION_LISTS and _engine and _engine.get('smiles'):",
        "if (_mid in PREFUSION_LISTS and _engine and _engine.get('smiles')\n            and _base_by_id.get(_mid, ([], [], 1.0))[2] < LIB_TAU):")
    run = run.replace("metric_fusion(_first, _engine['smiles'], metric_identity,",
        "consensus_backfill(_first, _engine['smiles'], BASE_TAIL_CANDIDATES.get(_mid, []), metric_identity,")
    run = run.replace('collapsed_source_votes=_collapsed', 'restored_tail_votes=_collapsed')
    run = run.replace("        _keys = [chem.score_key(s) or s for s in _candidates]",
        "        if not _collapsed:\n"
        "            _rows.append((_row.molecule_id, ';'.join(_original)))\n"
        "            _audit.append(dict(molecule_id=_mid, restored_tail_votes=0, changed=False, changed_top1=False))\n"
        "            continue\n"
        "        _keys = [chem.score_key(s) or s for s in _candidates]")
    nb['cells'].append(code_cell(prose('Consensus backfill hypothesis\n'
        'The winning fusion uses the preliminary 25-candidate first-engine list, while BASE_TAIL_CANDIDATES '
        'retains up to 60. A candidate below 25 therefore loses the first-engine vote even when the second '
        'engine finds it. Restore only these shared tail votes, without adding unsupported structures or '
        'compacting tail ranks. Keep strong-library molecules exactly as produced by the winning route. '
        'This overlay is our experiment; upstream engines and methods remain credited above. '
        'The final experiment_audit files compare against the completed 0.436 route from this run; '
        'candidate_audit files still describe the earlier inherited metric-safe pass. '
        'The prior metric-fusion audit is preserved separately. A no-op should not consume a submission.')
        + "import shutil\nshutil.copy2('experiment_audit.csv', 'metric_fusion_audit.csv')\n"
        + "shutil.copy2('experiment_audit.json', 'metric_fusion_audit.json')\n" + run))
    nb['metadata']['casmi_experiment'].update(overlay='consensus_backfill_on_metric_fusion',
        parent_user_reported_public_score=0.436, measured_score=None)
    validate(nb)
    return nb


if __name__ == '__main__':
    path = ROOT / 'experiments/casmi26_metric_fusion_consensus_backfill.ipynb'
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + '\n')
    print(path)
