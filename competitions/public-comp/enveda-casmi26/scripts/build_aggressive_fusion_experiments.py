'''Two independent overlays; run the repaired public adaptation before changing any predictions.'''
import copy
import json
from build_metric_safe_adaptation import build as baseline, code_cell
from build_flow_experiments import ROOT, prose, validate

METRIC_HELPER = '''
def metric_fusion(first, second, identity, alpha=0.6, offset=3.0, limit=40):
    scores, representatives = {}, {}
    collapsed = 0
    for candidates, weight in ((first, 1.0), (second, alpha)):
        seen = set()
        for candidate in candidates:
            key = identity(candidate)
            if key is None:
                continue
            if key in seen:
                collapsed += 1
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (offset + len(seen))
            representatives.setdefault(key, candidate)
    ordered = sorted(scores, key=lambda key: -scores[key])[:limit]
    return [representatives[key] for key in ordered], [scores[key] for key in ordered], collapsed
'''

GLOBAL_HELPER = '''
def global_glacier_order(candidates, evidence, offset=3.0, weight=1.0):
    import math
    values = []
    for candidate in candidates:
        value = evidence.get(candidate)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        values.append(float(value))
    if len(values) < 2:
        return None
    def z(values):
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
        return [(x - mean) / sd for x in values] if sd > 0 else None
    simulated = z(values)
    if simulated is None:
        return None
    prior = z([1.0 / (offset + rank) for rank in range(1, len(candidates) + 1)])
    return sorted(range(len(candidates)), key=lambda i: -(prior[i] + weight * simulated[i]))
'''

AUDIT = '''
_result = pd.DataFrame(_rows, columns=['molecule_id', 'smiles'])
assert list(_result.molecule_id) == list(_reference.molecule_id)
assert _result.molecule_id.is_unique and _result.smiles.notna().all()
assert _result.smiles.map(lambda value: 1 <= len(value.split(';')) <= 25).all()
_result.to_csv('submission.csv', index=False)
pd.DataFrame(_audit).to_csv('experiment_audit.csv', index=False)
_changed = sum(row['changed'] for row in _audit)
_report = dict(experiment=EXPERIMENT, changed_molecules=_changed,
               changed_top1=sum(row['changed_top1'] for row in _audit),
               reference_sha256=sha256('submission_reference.csv'),
               submission_sha256=sha256('submission.csv'), measured_score=None)
json.dump(_report, open('experiment_audit.json', 'w'), indent=2)
print(_report)
if not _changed:
    print('NO-OP: predictions match the reference; do not spend a second submission on this file.')
'''

METRIC_RUN = METRIC_HELPER + '''
EXPERIMENT = 'metric_identity_fusion'
_reference = pd.read_csv('submission.csv')
_reference.to_csv('submission_reference.csv', index=False)
_rows, _audit = [], []
for _row in _reference.itertuples(index=False):
    _mid = str(_row.molecule_id)
    _original = str(_row.smiles).split(';')
    _revised, _collapsed = list(_original), 0
    _engine = ENG.get(_mid)
    if _mid in PREFUSION_LISTS and _engine and _engine.get('smiles'):
        _first = [s for s in PREFUSION_LISTS[_mid] if s and s != 'CCO']
        _candidates, _scores, _collapsed = metric_fusion(_first, _engine['smiles'], metric_identity,
                                                       alpha=ALPHA, offset=KRR)
        _keys = [chem.score_key(s) or s for s in _candidates]
        _ice = ICE_SCORES.get(_mid, {}) if isinstance(ICE_SCORES, dict) else {}
        _gl = gl_of(_mid)
        if _ice and _gl:
            _order = gl_fuse.rerank_multi(_candidates, _keys, _scores,
                [_form2(s) for s in _candidates], [{}, _gl], [ICE_LAM, GL_LAM], top_n=len(_candidates))
            _candidates = [_candidates[i] for i in _order]
        if _candidates:
            _revised = safe_extend(_candidates[:25], _candidates[25:] + _original)
    _rows.append((_row.molecule_id, ';'.join(_revised)))
    _audit.append(dict(molecule_id=_mid, collapsed_source_votes=_collapsed,
        changed=_revised != _original, changed_top1=_revised[:1] != _original[:1]))
''' + AUDIT

GLOBAL_RUN = GLOBAL_HELPER + '''
EXPERIMENT = 'cross_formula_glacier'
_reference = pd.read_csv('submission.csv')
_reference.to_csv('submission_reference.csv', index=False)
_base_by_id = {str(mid): value for mid, value in BASE.items()}
_pc_by_id = {str(mid): value for mid, value in PC.items()}
_eligible = set(te.loc[(te.ionization_mode == 'positive') & (te.adduct == '[M+H]+'), 'molecule_id'].astype(str))
_global_candidates = {}
for _row in _reference.itertuples(index=False):
    _mid = str(_row.molecule_id)
    if _mid not in _eligible or _base_by_id.get(_mid, ([], [], 1.0))[2] >= LIB_TAU:
        continue
    _original = str(_row.smiles).split(';')
    _extra = (FUSION_OVERFLOW.get(_mid, []) + BASE_TAIL_CANDIDATES.get(_mid, [])
              + ENG.get(_mid, {}).get('smiles', []) + _pc_by_id.get(_mid, {}).get('pc', []))
    _candidates = safe_extend(_original, _extra, limit=125)
    if len({_form2(s) for s in _candidates}) >= 2:
        _global_candidates[_mid] = _candidates
_global_items = ice_fuse.build_ice_input(te, _global_candidates)
_global_items = [dict(item, spectra=[s for s in item['spectra'] if s.get('adduct') == '[M+H]+'])
                 for item in _global_items]
_global_items = [item for item in _global_items if item['spectra']]
_global_items.sort(key=lambda item: _base_by_id[item['mid']][2])
_global_scores = {}
if _global_items:
    _global_scores = gl_fuse.run_gl(GL_PKG, _global_items,
        workdir='/kaggle/working/gl_global_work', device='cuda',
        budget_s=min(1800, GL_BUDGET), site='/kaggle/working/ice_site',
        wheels=os.path.join(ICE_PKG, 'wheels'), extra_args=['--any-rdkit'])
_rows, _audit = [], []
for _row in _reference.itertuples(index=False):
    _mid = str(_row.molecule_id)
    _original = str(_row.smiles).split(';')
    _revised = list(_original)
    _candidates = _global_candidates.get(_mid, [])
    _evidence = _global_scores.get(_mid, {})
    _order = global_glacier_order(_candidates, _evidence, offset=KRR, weight=GL_LAM)
    if _order is not None:
        _revised = [_candidates[i] for i in _order[:25]]
    _rows.append((_row.molecule_id, ';'.join(_revised)))
    _audit.append(dict(molecule_id=_mid, proposed_candidates=len(_candidates),
        complete_scores=_order is not None, new_top25=sum(s not in _original for s in _revised),
        changed=_revised != _original, changed_top1=_revised[:1] != _original[:1]))
''' + AUDIT


def build():
    original = baseline()[0]
    notebooks = []
    for name, description, run in (
        ('Metric-Identity Fusion', 'One vote per official metric identity per engine; compact unique ranks before '
         'the 40-candidate fusion shortlist. Preserve the existing PubChem gate and same-formula GLACIER logic. '
         'This tests duplicate-vote bias, not a coefficient search. It can change top-1 and reduce the score.', METRIC_RUN),
        ('Cross-Formula GLACIER', 'Keep metric fusion unchanged. Rescore up to 125 existing unique candidates, '
         'including singleton formulas, only for weak-library molecules with positive [M+H]+ spectra. '
         'Rank globally by standardized reciprocal-rank prior plus GLACIER (existing weight). '
         'All proposed candidates must have finite scores; incomplete or constant evidence preserves the entire '
         'reference list. Strong-library and unsupported molecules are untouched. Cross-formula score '
         'calibration is unvalidated and can hurt top-1. Additional inference budget: at most 30 minutes.', GLOBAL_RUN)):
        nb = copy.deepcopy(original)
        nb['cells'][0] = code_cell(prose('CASMI26: ' + name + ' — Aggressive Experiment\n' + description + '\n'
            'Run the complete repaired @obstacledeveloper adaptation first, then this independent overlay. '
            'submission_reference.csv is the completed baseline from the same run; experiment_audit.csv/json '
            'record changes and file hashes. This is not stacked with the other experiment. '
            'Same 17 source inputs and GPU environment; no additional dataset required. '
            'No new public score has been measured. A visible smoke run is not validation.'))
        nb['cells'].insert(2, code_cell(prose('Experiment provenance and reasoning\n'
            'Direct starting point: @obstacledeveloper, linked in Credits & Attribution below. '
            'The upstream engine/fusion/simulator code is substantially reused, not claimed as original. '
            'Our contribution here is the separately audited overlay described above. '
            'Previous analog expansion, top-1 shielding, adduct-specific merging and single-pass variants '
            'did not improve our confirmed 0.420 result. These experiments test structural ranking changes '
            'instead of repeating those parameter changes. Metric identity uses the inherited official '
            'canonicalization and heavy-atom composition helpers. GLACIER tooling: '
            'https://github.com/coleygroup/ms-pred ; research: https://arxiv.org/abs/2606.29161 . '
            'Global cross-formula ranking is our unvalidated hypothesis, not a claim made by that paper. '
            'Public leaderboard feedback does not establish private-set generalization.')))
        for cell in nb['cells']:
            value = ''.join(cell['source'])
            if 'FUSION_OVERFLOW = {}' in value:
                value = value.replace('FUSION_OVERFLOW = {}',
                    "PREFUSION_LISTS = {str(r.molecule_id): str(r.smiles).split(';') for r in pd.read_csv('submission.csv').itertuples(index=False)}\nFUSION_OVERFLOW = {}", 1)
                cell['source'] = value.splitlines(keepends=True)
        nb['cells'].append(code_cell(prose('Independent experimental overlay\n' + description) + run))
        nb['metadata']['casmi_experiment'].update(overlay=name, measured_score=None,
            validation='Syntax and focused synthetic tests; full Kaggle inference pending')
        validate(nb)
        notebooks.append(nb)
    return notebooks


if __name__ == '__main__':
    for name, nb in zip(('casmi26_metric_identity_fusion.ipynb', 'casmi26_cross_formula_glacier.ipynb'), build()):
        path = ROOT / 'experiments' / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path)
