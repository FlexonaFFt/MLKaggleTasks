'''Independent retrieval and joint-ranking overlays on confirmed E006 0.440.'''
import json
from build_ice_fallback_versions import build as e006_build
from build_metric_safe_adaptation import code_cell
from build_flow_experiments import ROOT, prose, validate

HELPERS = '''
def polarity_union(baseline, positive, negative, identity, offset=3., limit=25):
    scores, representative = {}, {}
    for candidates, weight in ((baseline, 1.), (positive, .5), (negative, .5)):
        seen = set()
        for smile in candidates:
            key = identity(smile)
            if key is None or key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.) + weight / (offset + len(seen))
            representative.setdefault(key, smile)
    order = sorted(scores, key=lambda key: -scores[key])[:limit]
    return [representative[key] for key in order]

def joint_common_score(fingerprint, popularity, support, pop_weight=.25):
    import numpy as np
    f, p, s = (np.asarray(value, float) for value in (fingerprint, popularity, support))
    assert f.ndim == p.ndim == s.ndim == 1 and len(f) == len(p) == len(s)
    if not (np.isfinite(f).all() and np.isfinite(p).all() and np.isfinite(s).all()):
        raise ValueError('Common ranking requires complete finite feature coverage')
    def z(values):
        sd = values.std()
        return (values - values.mean()) / sd if sd > 1e-9 else np.zeros(len(values))
    return z(f) + pop_weight * p + z(s)
'''

WORKER = '''
import sys, os, json, glob, time
import numpy as np, pandas as pd

def checkpoint(value, path):
    with open(path + '.tmp', 'w') as handle:
        json.dump(value, handle, allow_nan=False)
    os.replace(path + '.tmp', path)

def spectra_for(frame):
    spectra = []
    for row in frame.itertuples(index=False):
        ce = np.atleast_1d(np.asarray(row.collision_energy_ev, float)) if row.collision_energy_ev is not None else []
        spectra.append(dict(mz=np.asarray(row.ms2_mzs, float), it=np.asarray(row.ms2_normalized_intensities, float),
            mode=1 if row.ionization_mode == 'positive' else -1, adduct=row.adduct, prec=float(row.precursor_mz),
            ce=float(np.mean(ce)) if len(ce) else np.nan, ce_n=len(ce) if len(ce) else 1))
    return spectra[:16]

if __name__ == '__main__':
    MODE, CODE, MODELS, POOL, PC, TEST, SPEC, OUT = sys.argv[1:]
    sys.path.insert(0, CODE)
    import torch
    import probe_core2 as pc
    from casmi import fpnet, chem
    spec = json.load(open(SPEC))
    frame = pd.read_parquet(TEST)
    frame['nm'] = chem.neutral_mass(frame.precursor_mz.values.astype(float), frame.adduct.values)
    files = sorted(glob.glob(os.path.join(MODELS, 'fpnet_*.pt')))
    assert len(files) == 2
    bank = fpnet.ModelBank(files, device='cuda' if torch.cuda.is_available() else 'cpu')
    bits = np.load(os.path.join(POOL, 'fp_bits.npy'))
    output, tasks = {}, []
    if MODE == 'joint':
        pc.init_worker(PC, os.path.join(POOL, 'fp_bits.npy'), os.path.join(POOL, 'pool_meta.parquet'), CODE)
    for mid, sub in frame.groupby('molecule_id', sort=False):
        mid = str(mid)
        if mid not in spec:
            continue
        try:
            spectra = spectra_for(sub)
            finite_mass = sub.nm.values[np.isfinite(sub.nm.values)]
            if not len(finite_mass):
                continue
            target = float(np.median(finite_mass))
            if MODE == 'dual':
                if {s['mode'] for s in spectra} != {-1, 1}:
                    continue
                for mode in (1, -1):
                    z = pc.molecule_logits(bank, [s for s in spectra if s['mode'] == mode])
                    if z is None or not np.isfinite(z).all():
                        raise ValueError('Incomplete polarity logits')
                    tasks.append((mid + ('|positive' if mode == 1 else '|negative'), target, z))
            else:
                z = pc.molecule_logits(bank, spectra)
                if z is None or len(z) != len(bits) or not np.isfinite(z).all():
                    raise ValueError('Invalid common logits')
                candidates = spec[mid]['smiles']
                tol = target * pc.PPM * 1e-6
                a = int(np.searchsorted(pc._W['mass'], target - tol, 'left'))
                window = pc._window(target)
                pops = (np.asarray(pc._W['ls'][a:a+len(window)], float) +
                        np.asarray(pc._W['lp'][a:a+len(window)], float))
                pop_by_smile = dict(zip(window, pops))
                fits, popularity = [], []
                for smile, pool_pop in zip(candidates, spec[mid]['pool_pop']):
                    fingerprint = chem.raw_fingerprint(smile)
                    if fingerprint is None:
                        raise ValueError('Missing candidate fingerprint')
                    fits.append(float(fingerprint[bits].astype(np.float64) @ z.astype(np.float64)))
                    popularity.append(float(pool_pop) if pool_pop is not None else float(pop_by_smile.get(smile, 0.)))
                if not np.isfinite(fits + popularity).all():
                    raise ValueError('Non-finite common evidence')
                output[mid] = dict(smiles=candidates, fingerprint=fits, popularity=popularity)
                checkpoint(output, OUT)
        except Exception as error:
            print('RETRIEVAL FAILED', mid, repr(error), flush=True)
    del bank
    torch.cuda.empty_cache()
    if MODE == 'dual' and tasks:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(4, initializer=pc.init_worker,
                initargs=(PC, os.path.join(POOL, 'fp_bits.npy'), os.path.join(POOL, 'pool_meta.parquet'), CODE)) as workers:
            for tag, smiles, scores, keys, diagnostic in workers.imap_unordered(pc.probe_one, tasks, chunksize=1):
                mid, polarity = tag.rsplit('|', 1)
                output.setdefault(mid, {})[polarity] = dict(smiles=smiles, error=False)
                checkpoint(output, OUT)
    checkpoint(output, OUT)
    print('RETRIEVAL OUTPUT', MODE, len(output), flush=True)
'''

SETUP = '''
import shutil
from pathlib import Path
shutil.copy2('submission.csv', 'submission_e006_control.csv')
_control = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
_base = {str(mid): value for mid, value in BASE.items()}
_pc = {str(mid): value for mid, value in PC.items()}
_spec, _support = {}, {}
_pool_index = {str(key): i for i, key in enumerate(P.key)} if MODE == 'joint' else {}
_polarities = te.groupby('molecule_id').ionization_mode.agg(lambda values:set(values)).to_dict()
for row in _control.itertuples(index=False):
    mid = str(row.molecule_id)
    if mid not in _base or _base[mid][2] >= LIB_TAU:
        continue
    if MODE == 'dual':
        modes = next((value for key, value in _polarities.items() if str(key) == mid), set())
        if modes == {'positive', 'negative'} and _pc.get(mid, {}).get('pc'):
            _spec[mid] = {}
    else:
        first = BASE_TAIL_CANDIDATES.get(mid, [])
        second = ENG.get(mid, {}).get('smiles', [])
        smiles = safe_extend(str(row.smiles).split(';'), first + second + _pc.get(mid, {}).get('pc', []), limit=150)
        supported, support, _ = metric_fusion(first, second, metric_identity, alpha=ALPHA, offset=KRR, limit=150)
        support_by_key = dict(zip(map(metric_identity, supported), support))
        _support[mid] = [float(support_by_key.get(metric_identity(s), 0.)) for s in smiles]
        pop = []
        for smile in smiles:
            index = _pool_index.get(str(chem.score_key(smile)))
            pop.append(float(POOL_POP[index]) if index is not None else None)
        _spec[mid] = dict(smiles=smiles, pool_pop=pop)
Path('retrieval_experiment_spec.json').write_text(json.dumps(_spec, allow_nan=False))
Path('pubchem_experiment_worker.py').write_text(WORKER_SOURCE)
_retrieved = {}
if _spec:
    try:
        Path('retrieval_experiment_output.json').unlink(missing_ok=True)
        torch.cuda.empty_cache()
        result = subprocess.run([sys.executable, '/kaggle/working/pubchem_experiment_worker.py', MODE,
            V3CODE, V3MOD, POOL_DIR, PC_DIR, os.path.join(COMP, 'test.parquet'),
            '/kaggle/working/retrieval_experiment_spec.json', '/kaggle/working/retrieval_experiment_output.json'],
            capture_output=True, text=True, timeout=4*3600)
        print(result.stdout[-2500:], result.stderr[-1500:])
    except subprocess.TimeoutExpired:
        print('Retrieval budget exhausted: use only complete checkpointed molecules.')
    if Path('retrieval_experiment_output.json').is_file():
        _retrieved = json.loads(Path('retrieval_experiment_output.json').read_text())
'''

RUN = '''
_rows, _audit = [], []
for row in _control.itertuples(index=False):
    mid, original = str(row.molecule_id), str(row.smiles).split(';')
    revised, reason = list(original), 'unchanged'
    evidence = _retrieved.get(mid)
    if mid in _spec and evidence:
        if MODE == 'dual' and evidence.get('positive', {}).get('smiles') and evidence.get('negative', {}).get('smiles'):
            p = _pc[mid]
            pc_smiles = polarity_union(p['pc'], evidence['positive']['smiles'], evidence['negative']['smiles'], metric_identity, KRR)
            prefusion = PREFUSION_LISTS[mid]
            promoted = bool(prefusion and p['pc'] and prefusion[0] == p['pc'][0])
            if promoted:
                pc_smiles = [p['pc'][0]] + [s for s in pc_smiles if metric_identity(s) != metric_identity(p['pc'][0])]
                pc_smiles = pc_smiles[:25]
            best = p['best_pool_fz']
            rel = ((p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - best) if np.isfinite(best) else 1e9
            slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
            base_smiles, base_keys, _ = _base[mid]
            fn = promote if promoted else merge
            first = fn(base_smiles, base_keys, pc_smiles, [chem.score_key(s) for s in pc_smiles], slots)
            candidates, scores, _ = metric_fusion(first, ENG[mid]['smiles'], metric_identity, alpha=ALPHA, offset=KRR)
            reason = 'dual_retrieval'
        elif MODE == 'joint' and evidence.get('smiles') == _spec[mid]['smiles']:
            scores_array = joint_common_score(evidence['fingerprint'], evidence['popularity'], _support[mid])
            order = np.argsort(-scores_array, kind='stable')[:40]
            candidates = [evidence['smiles'][i] for i in order]
            scores = [float(scores_array[i]) for i in order]
            reason = 'joint_common_ranking'
        else:
            candidates = []
        if candidates:
            keys, forms = [chem.score_key(s) or s for s in candidates], [_form2(s) for s in candidates]
            ice, gl = ICE_SCORES.get(mid, {}), gl_of(mid)
            if ice and gl:
                order = gl_fuse.rerank_multi(candidates, keys, scores, forms, [{},gl], [ICE_LAM,GL_LAM], top_n=len(candidates))
                candidates = [candidates[i] for i in order]
            elif ice and not gl:
                order = ice_fuse.rerank(candidates, keys, scores, forms, ice, lam=ICE_LAM, top_n=len(candidates))
                candidates = [candidates[i] for i in order]
            revised = safe_extend(candidates[:25], candidates[25:] + original)
    _rows.append((row.molecule_id, ';'.join(revised)))
    _audit.append(dict(molecule_id=mid, branch=reason, changed=revised != original,
        changed_top1=revised[:1] != original[:1], new_candidates=sum(metric_identity(s) not in set(map(metric_identity, original)) for s in revised)))
_output = pd.DataFrame(_rows, columns=['molecule_id','smiles'])
assert list(_output.molecule_id) == list(_control.molecule_id) and _output.molecule_id.is_unique
assert _output.smiles.map(lambda value: 1 <= len(value.split(';')) <= 25).all()
_output.to_csv('submission.csv', index=False)
pd.DataFrame(_audit).to_csv('retrieval_experiment_audit.csv', index=False)
_report = dict(mode=MODE, eligible=len(_spec), checkpointed=len(_retrieved),
    changed_rows=sum(r['changed'] for r in _audit), changed_top1=sum(r['changed_top1'] for r in _audit),
    control_sha256=sha256('submission_e006_control.csv'), submission_sha256=sha256('submission.csv'), measured_score=None)
Path('retrieval_experiment_audit.json').write_text(json.dumps(_report, indent=2))
print(_report)
if not _report['changed_rows']:
    print('NO-OP versus E006: do not spend a separate submission on identical predictions.')
'''


def build():
    notebooks = []
    for mode, title, explanation in (
        ('dual', 'Dual-Polarity PubChem Retrieval',
         'For weak-library molecules with BOTH positive and negative spectra, obtain separate A+B '
         'FPNet logits and PubChem retrieval lists. Combine their ranks with the existing combined '
         'PubChem list: weights 0.5 positive, 0.5 negative, 1.0 baseline. Deduplicate by official metric '
         'identity. Preserve the original slot gate and pin the original promoted PubChem candidate '
         'when promotion was active. Rebuild metric fusion and the original GLACIER-else-ICE policy. '
         'Single-polarity, strong-library and incomplete retrieval molecules are untouched. '
         'New candidates may lack simulator evidence: this is an aggressive coverage tradeoff.'),
        ('joint', 'Joint Pool + PubChem Common-Evidence Ranking',
         'For weak-library molecules, combine existing E006, base top-60, Engine2 top-40 and PubChem '
         'top-25 candidates, deduplicated by official metric identity. Score EVERY candidate using '
         'the SAME A+B FPNet logits and selected fingerprint bits. Rank by z(common fingerprint fit) '
         '+ 0.25 * existing log-popularity + z(metric reciprocal-rank support from the two pool engines). '
         'Never add the old channel scores on incompatible scales. Missing popularity is zero; '
         'missing fingerprint coverage preserves E006. Replace fixed PubChem slots with this common '
         'ranking, then the original same-formula simulator rerank and final top-25. '
         'This is a heuristic, not a trained/calibrated ranker; it can strongly change top-1.')):
        nb = e006_build()[0]
        nb['cells'][0] = code_cell(prose('CASMI26: ' + title + '\n'
            'Independent experiment on E006, user-confirmed public MRR 0.440. ' + explanation + '\n'
            'No score claimed for this new version. Same 17 sources, Python 3.12, T4 GPU. '
            'Additional FPNet/PubChem processing runs in an isolated subprocess, capped at four hours. '
            'No new dataset, new simulator pass, window80 or spectral-quality overlay.'))
        nb['cells'].append(code_cell(prose('Experiment rationale and attribution\n' + explanation + '\n'
            'Our contribution is this retrieval/common-evidence experiment. @ahmedberatozer supplies '
            'the reused FPNet/PubChem implementation and candidate resources; @dmitriigluzdov supplies '
            'the inherited popularity prior. Direct E006 source, @samanyu1808 fallback policy, '
            '@prvsiyan and all upstream authors/links remain credited above. The inherited worker '
            'CORE + V17_PATCH is reused, not claimed as original. Same-run E006 output is retained '
            'byte-for-byte. Check the paired audit; a visible smoke no-op does not prove hidden inactivity.')
            + HELPERS + f"\nMODE = {mode!r}\nWORKER_SOURCE = {WORKER!r}\n" + SETUP + RUN))
        nb['metadata']['casmi_experiment'].update(overlay=mode + '_pubchem', measured_score=None,
            user_reported_baseline_score=.440, validation='Syntax and synthetic overlay checks; Kaggle run pending')
        validate(nb)
        notebooks.append(nb)
    return notebooks


if __name__ == '__main__':
    for name, nb in zip(('casmi26_dual_polarity_pubchem.ipynb', 'casmi26_joint_pool_pubchem_ranking.ipynb'), build()):
        path = ROOT / 'experiments' / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path)
