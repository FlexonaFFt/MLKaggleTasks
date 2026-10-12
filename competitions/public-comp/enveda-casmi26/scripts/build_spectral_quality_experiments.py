'''Independent scoring-only experiments on user-confirmed E006 0.440.'''
import json
import hashlib
from build_ice_fallback_versions import build as e006_build, SOURCE
from build_metric_safe_adaptation import code_cell, comments
from build_flow_experiments import ROOT, prose, replace, validate

SCORING = '''
import numpy as np

def quality_weights(spectra):
    qualities, counts = [], []
    for sp in spectra:
        mz, it = np.asarray(sp['mz'], float), np.asarray(sp['it'], float)
        keep = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz > 0) & (mz <= float(sp['prec']) + 2)
        values = it[keep]
        if len(values) and values.sum() > 0:
            p = values / values.sum()
            effective = np.exp(-np.sum(p * np.log(p)))
        else:
            effective = 1.0
        qualities.append(effective)
        count = sp.get('base_peak_intensity')
        counts.append(float(count) if count is not None and np.isfinite(count) and count > 0 else np.nan)
    if not qualities:
        return np.zeros(0)
    med_quality = max(float(np.median(qualities)), 1.)
    good_counts = np.asarray(counts)[np.isfinite(counts)]
    med_count = float(np.median(good_counts)) if len(good_counts) else 1.
    result = []
    for sp, quality, count in zip(spectra, qualities, counts):
        intensity = np.sqrt(count / med_count) if np.isfinite(count) else 1.
        energies = max(1, len(sp.get('ce_ev') or []))
        result.append(float(np.clip(np.sqrt(quality / med_quality) * intensity / np.sqrt(energies), .5, 2.)))
    return np.asarray(result)

def precursor_mask(mz, intensity, precursor):
    mz, intensity = np.asarray(mz), np.asarray(intensity)
    tolerance = max(.02, float(precursor) * 20e-6)
    keep = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0) & (np.abs(mz - precursor) > tolerance)
    return mz[keep], intensity[keep]

def install(scope, mode):
    original_plan = scope['molecule_plan']
    metadata = {}
    def plan(mol):
        result = original_plan(mol)
        covered = [sp for sp in mol.get('spectra', []) if scope['is_covered'](sp)]
        assert len(result) == len(covered)
        metadata[id(result)] = (covered, quality_weights(covered))
        return result
    def score(plan, candidates, cache):
        spectra, weights = metadata[id(plan)]
        output, summed = {}, {}
        for candidate in candidates:
            similarities, used_weights = [], []
            for j, (qm, qp, adduct, instrument, energy) in enumerate(plan):
                if np.isfinite(energy):
                    prediction = cache.get((candidate, adduct, instrument, energy))
                else:
                    key = (candidate, adduct, instrument)
                    if key not in summed:
                        parts = [cache.get((candidate, adduct, instrument, ce)) for ce in scope['SUM_CE']]
                        summed[key] = scope['merge'](parts) if all(p is not None for p in parts) else None
                    prediction = summed[key]
                if prediction is None:
                    continue
                pm, pp = scope['prep_query'](prediction[0], prediction[1],
                    scope['FLOOR'], scope['TOPK'], scope['POWER'], scope['ENTW'])
                query_m, query_p = qm, qp
                if mode == 'precursor_masked':
                    sp = spectra[j]
                    mz, it = precursor_mask(sp['mz'], sp['it'], sp['prec'])
                    predicted_mz, predicted_it = precursor_mask(prediction[0], prediction[1], sp['prec'])
                    if len(mz) >= 5 and len(predicted_mz) >= 5:
                        masked_sp = dict(sp, mz=mz.tolist(), it=it.tolist())
                        mm, mp = scope['prep_spectrum'](masked_sp)
                        pmask, pweights = scope['prep_query'](predicted_mz, predicted_it,
                            scope['FLOOR'], scope['TOPK'], scope['POWER'], scope['ENTW'])
                        if len(mm) >= 5 and len(pmask) >= 5:
                            query_m, query_p, pm, pp = mm, mp, pmask, pweights
                similarity = scope['entropy_sim'](query_m, query_p, pm, pp, scope['TOL'], scope['PPM'])
                if np.isfinite(similarity):
                    similarities.append(similarity)
                    used_weights.append(weights[j] if mode == 'quality_weighted' else 1.)
            output[candidate] = float(np.average(similarities, weights=used_weights)) if similarities else None
        return output
    scope['molecule_plan'], scope['score_molecule'] = plan, score
'''

RERUN = r'''
import ast, shutil
def variant_package(pkg, runner_name):
    destination = Path('/kaggle/working') / (MODE + '_' + runner_name.replace('.py', ''))
    destination.mkdir(exist_ok=True)
    for entry in Path(pkg).iterdir():
        if entry.name == runner_name:
            continue
        link = destination / entry.name
        if not link.exists():
            link.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
    code = (Path(pkg) / runner_name).read_text()
    tree = ast.parse(code)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert 'np.mean(sims)' in ast.unparse(functions['score_molecule']), 'Runner aggregation changed; review before patching'
    assert 'for _, _, ad, ins, ceb in plan' in ast.unparse(functions['molecule_jobs'])
    anchor = "if __name__ == '__main__':"
    assert code.count(anchor) == 1
    code = code.replace(anchor, "from spectral_scoring_variant import install\ninstall(globals(), " + repr(MODE) + ")\n\n" + anchor)
    (destination / runner_name).write_text(code)
    (destination / 'spectral_scoring_variant.py').write_text(SCORING_VARIANT)
    return str(destination)

def keep_complete(original, revised):
    result, accepted = dict(original), 0
    for mid, scores in original.items():
        finite = {s for s, value in scores.items() if value is not None and np.isfinite(value)}
        candidate = revised.get(mid, {})
        if finite and all(candidate.get(s) is not None and np.isfinite(candidate[s]) for s in finite):
            result[mid] = {s: float(candidate[s]) if s in finite else value for s, value in scores.items()}
            accepted += 1
    return result, accepted

def add_intensity(items):
    rows = {}
    for row in te.itertuples(index=False):
        key = (str(row.molecule_id), str(row.adduct), float(row.precursor_mz), tuple(row.ms2_mzs))
        rows[key] = float(getattr(row, 'base_peak_intensity', np.nan))
    for item in items:
        for sp in item['spectra']:
            key = (item['mid'], sp['adduct'], sp['prec'], tuple(sp['mz']))
            count = rows.get(key, np.nan)
            sp['base_peak_intensity'] = count if np.isfinite(count) else None
    return items

_gl_input = json.loads(Path('gl_work/gl_in.json').read_text())
_ice_input = json.loads(Path('ice_work/ice_in.json').read_text())
_gl_input = [item for item in _gl_input if gl_of(item['mid'])]
_ice_input = [item for item in _ice_input if not gl_of(item['mid']) and ICE_SCORES.get(item['mid'])]
if MODE == 'quality_weighted':
    _gl_input, _ice_input = add_intensity(_gl_input), add_intensity(_ice_input)
_new_gl, _new_ice = {}, {}
if _gl_input:
    _pkg = variant_package(GL_PKG, 'gl_runner.py')
    _new_gl = gl_fuse.run_gl(_pkg, _gl_input, workdir='/kaggle/working/' + MODE + '_gl',
        device='cuda', budget_s=GL_BUDGET, site='/kaggle/working/ice_site',
        wheels=os.path.join(ICE_PKG, 'wheels'), extra_args=['--any-rdkit'])
if _ice_input:
    _pkg = variant_package(ICE_PKG, 'ice_runner.py')
    _new_ice = ice_fuse.run_ice(_pkg, _ice_input, workdir='/kaggle/working/' + MODE + '_ice',
        device='cuda', budget_s=ICE_BUDGET, site='/kaggle/working/ice_site', extra_args=['--any-rdkit'])
VARIANT_GL, _ng = keep_complete(GL_SCORES, _new_gl)
VARIANT_ICE, _ni = keep_complete(ICE_SCORES, _new_ice)
Path('spectral_variant_coverage.json').write_text(json.dumps(dict(mode=MODE,
    gl_recomputed_molecules=_ng, ice_recomputed_molecules=_ni, gl_input_molecules=len(_gl_input),
    ice_input_molecules=len(_ice_input)), indent=2))
print('Complete variant score coverage', MODE, _ng, _ni)
'''


def build():
    base = e006_build()[0]
    result = []
    for mode, title, explanation in (
        ('quality_weighted', 'Quality-Weighted Spectral Evidence',
         'Replace the unweighted mean of per-spectrum entropy similarity with bounded quality weights '
         '[0.5, 2.0]. Quality combines effective peak count, relative raw base-peak intensity and '
         'a square-root penalty for merged collision energies. Missing raw intensity is neutral. '
         'Weights are candidate-independent, calculated among covered spectra of each molecule. '
         'No peak filtering or predictor change. These quality proxies are unvalidated and may amplify noise.'),
        ('precursor_masked', 'Precursor-Masked Simulator Matching',
         'Remove peaks within max(0.02 Da, 20 ppm) of the measured precursor from BOTH observed and '
         'predicted spectra before the original cleaning, normalization and entropy weighting. '
         'Apply masking only when BOTH prepared spectra retain at least five peaks; otherwise '
         'use original similarity for that candidate/spectrum. Keep the unweighted mean. '
         'Precursor peaks can be informative; removing them may hurt discrimination.')):
        nb = json.loads(json.dumps(base))
        nb['cells'][0] = code_cell(prose('CASMI26: ' + title + '\n'
            'Independent experiment on E006, confirmed public MRR 0.440 by the user. '
            + explanation + '\nRun the full E006 control first, then repeat forward simulation with only '
            'the scoring functions changed. Preserve window=40, fusion weights, candidate sources, '
            'PubChem gates and GLACIER-else-ICEBERG policy. ICEBERG is repeated only where GLACIER '
            'was absent. Incomplete variant coverage keeps original scores per molecule. '
            'Same inputs, Python 3.12 and T4 GPU. Runtime increases by up to the inherited simulator '
            'budgets plus startup overhead; no measured score for this experiment.'))
        original = ''.join(json.loads(SOURCE.read_text())['cells'][-1]['source'])
        original = replace(original, '        gl = scope["gl_of"](mid)\n',
            '        gl = scope["gl_of"](mid)\n'
            '        if ice_fallback:\n'
            '            ice = scope["VARIANT_ICE"].get(mid, ice)\n'
            '            gl = scope["VARIANT_GL"].get(mid, gl)\n')
        original = replace(original, "'experiment': 'E006', 'window': 40, 'ice_fallback': True",
                           f"'experiment': '{mode}', 'window': 40, 'ice_fallback': True")
        original = replace(original, "'helper_source_sha256':", "'inherited_helper_source_sha256':")
        from build_simulator_evidence_experiments import COMPARE
        code = (prose('Scoring-only overlay and attribution\n' + explanation + '\n'
            'Runner/model authors: @ahmedberatozer, ICEBERG/GLACIER tooling '
            'https://github.com/coleygroup/ms-pred . @samanyu1808 fallback policy and all inherited '
            'engine/metric authors remain credited above. Our contribution is the scoring override, '
            'not the predictors. Upstream runners are checked before patching; model/job generation '
            'and cached prediction interpretation stay unchanged. Save an exact E006 control for comparison.')
            + f"MODE = EXPERIMENT = {mode!r}\nSCORING_VARIANT = {SCORING!r}\n" + RERUN
            + "shutil.copy2('submission.csv', 'submission_e006_control.csv')\n"
            + "shutil.copy2('experiment_verification.json', 'e006_verification.json')\n"
            + "shutil.copy2('experiment_audit_factorial.csv', 'e006_audit.csv')\n"
            + "shutil.copy2('submission_e004_control.csv', 'submission.csv')\n"
            + comments(original) + COMPARE)
        nb['cells'].append(code_cell(code))
        nb['metadata']['casmi_experiment'].update(overlay=mode, measured_score=None,
            user_reported_baseline_score=.440, scoring_override_sha256=hashlib.sha256(SCORING.encode()).hexdigest())
        validate(nb)
        result.append(nb)
    return result


if __name__ == '__main__':
    for name, nb in zip(('casmi26_quality_weighted_spectral_evidence.ipynb',
                         'casmi26_precursor_masked_matching.ipynb'), build()):
        path = ROOT / 'experiments' / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path)
