#!/usr/bin/env python
"""GLACIER isomer re-scorer -- self-contained subprocess runner for the CASMI26 Kaggle notebook.

    python gl_runner.py <package_dir> <in.json> <out.json> [--device cuda] [--budget SEC] [--site DIR|none]
                        [--wheels DIR] ...

Fork of casmi26-iceberg/ice_runner.py: SAME CLI, input format, job construction (adduct / instrument token / CE
bucket), scoring (ice_ex_mean definition), chunking, time budget, checkpointing, meta JSON and failure behaviour.
Only model loading + prediction are replaced by GLACIER (ms-pred glacier.JointModel, MassSpecGym checkpoint, MIT).

Runs in its OWN process with its own sys.path (package shim + ms_pred src + an offline --target install of the
casmi26-iceberg wheels incl. RDKit 2025.03), so the notebook process keeps RDKit 2026.03 and never imports the
torch-DGL / LinSAT stand-ins.  --site DIR: the site dir the ICE runner already filled (marker .ice_site_ok) is reused
as is; otherwise the runner installs the wheels itself from --wheels DIR (default: <package>/wheels, env GL_WHEELS,
then /kaggle/input/**/casmi26-iceberg/**/wheels); no wheels and no usable site -> status error, all-null output.

in.json  : [{"mid": str, "cands": [smiles, ...],
             "spectra": [{"mz": [...], "it": [...], "prec": float, "adduct": "[M+H]+", "mode": "positive" | +1 | -1,
                          "instrument": raw instrument_type string (optional; default -> QTOF token),
                          "ce_ev": [...], "ce_orig": str, "ce_units": "eV" | "NCE"     (raw test-parquet fields), or
                          "ce": float | null   (already-resolved CE, used only when ce_ev / ce_units are absent)}]}]
out.json : {mid: {smiles: gl_ex_mean or null}}   (null = no covered spectrum / prediction failed / not reached)
out.json.meta.json : run statistics (status, timings, versions, errors).

Score = the A1 panel definition `gl_ex_mean` (work/analysis/paradigm/program/a1_glacier: glacier_load.py,
glacier_predict.py, gl_score.py = ice_ft/panel_score.py = the ice_ex_mean definition):
  covered spectrum : mode +1 and adduct in {[M+H]+, [M+Na]+}
  instrument token : 'Orbitrap' if instr_family(instrument_type) == 1 else 'QTOF'
  CE               : units == 'NCE' -> mean of the numbers in collision_energy_orig, else mean(collision_energy_ev)
                     (NaN if empty); bucket = np.round(ce / 5) * 5, min 5; NaN -> 'sum' prediction (CE 20+40+60 merged)
  molecule         : predict_smis_joint.prepare_entry canonicalisation (valid atoms only, RemoveHs, <= 100 heavy
                     atoms, MolToSmiles); rejected -> null
  prediction       : TreeProcessor.featurize_tree -> IntenDataset.collate_fn -> JointModel.predict_inten_frag_batch
                     (collision_engs = CE bucket, instrument = token, adduct as given, precursor 0) -> drop rows with
                     intensity or m/z <= 0 -> merge identical m/z (round 4) by MAX -> top 100 -> sort by m/z ->
                     max 1 (float32)
  query            : raw peaks -> clean_peaks(prec, floor 0.001, top 512, <= prec + 2) (library cache, float32) ->
                     clean_peaks again (score.py) -> prep_query(floor 0.002, top 256, power 1, entropy weighting)
  similarity       : engine entropy_sim, 0.01 Da / 20 ppm;   gl_ex_mean = mean over the molecule's covered spectra
                     that have a prediction (NaN sims skipped).
Batching: jobs sorted by (heavy atoms, canonical SMILES, adduct, instrument, CE); batch = min(bs, bs*1600 // N^2)
(N = max heavy atoms of the next bs jobs) = glacier_load.batch_size_for at bs 32.
Robustness: the output file is written (all null) before anything heavy happens and rewritten after every chunk, so a
crash, a kill or the time budget always leaves a valid out.json; the process exits 0 whenever out.json was written.
Added vs ice_runner: fail-fast -- if the first FAILFAST_N predictions ALL fail (systematic, e.g. device) the run stops
with status error instead of splitting every batch down to single jobs until the budget is gone.
"""
import os, sys, json, time, re, math, argparse, traceback, subprocess, glob, types, importlib
from collections import OrderedDict
sys.dont_write_bytecode = True          # never write __pycache__ into the (read-only on Kaggle) package dir

COV = ('[M+H]+', '[M+Na]+')
SUM_CE = (20.0, 40.0, 60.0)
FLOOR, TOPK, POWER, ENTW, TOL, PPM = 0.002, 256, 1.0, True, 0.01, 20.0
RDKIT_REQ = '2025.03'
ATOM_BUDGET_PER_BS = 40 * 40            # glacier_load.batch_size_for: budget = 40 * 40 * 32 at bs_max 32
FAILFAST_N = 64                         # this many predictions, ALL failed -> abort (status error) instead of burning the budget


# ----------------------------------------------------------------------------------------------------------- io ----
def write_json(path, obj):
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, allow_nan=False)
    os.replace(tmp, path)


class Log:
    def __init__(self, quiet=False):
        self.quiet = quiet; self.lines = []; self.t0 = time.time()

    def __call__(self, *a):
        s = f'[gl {time.time() - self.t0:7.1f}s] ' + ' '.join(str(x) for x in a)
        self.lines.append(s)
        if not self.quiet:
            print(s, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------- environment -----
def find_wheels(pkg, explicit=None):
    """Wheel dir: --wheels, <pkg>/wheels, env GL_WHEELS, then the casmi26-iceberg dataset on Kaggle."""
    cands = [explicit, os.path.join(pkg, 'wheels'), os.environ.get('GL_WHEELS')]
    cands += sorted(glob.glob('/kaggle/input/**/casmi26-iceberg/**/wheels', recursive=True), key=len)
    for d in cands:
        if d and os.path.isdir(d) and (glob.glob(os.path.join(d, '*.whl.ice')) or glob.glob(os.path.join(d, '*.whl'))):
            return d
    return None


def install_site(wdir, site, log):
    """= ice_runner.install_site with the wheel dir as argument: offline install of <wdir>/*.whl.ice (= *.whl renamed
    so that the notebook's own `rdkit-*.whl` glob can never pick them up) into `site` with pip --no-index --no-deps
    --target; idempotent via the SAME marker file as ice_runner (a site the ICE runner filled is reused as is)."""
    import shutil
    mark = os.path.join(site, '.ice_site_ok')
    src = sorted(glob.glob(os.path.join(wdir, '*.whl.ice')) + glob.glob(os.path.join(wdir, '*.whl'))) if wdir else []
    if not src:
        if os.path.exists(mark):
            log('no wheel dir found; reusing the existing site', site)
            return
        raise RuntimeError(f'no wheels found (--wheels / <pkg>/wheels / GL_WHEELS / casmi26-iceberg) and no ready site {site}')
    want = '\n'.join(os.path.basename(w) for w in src)
    if os.path.exists(mark) and open(mark).read() == want:
        log('site ready (marker matches):', site)
        return
    tmpd = os.path.join(site, '.wheels')
    os.makedirs(tmpd, exist_ok=True)
    whl = []
    for w in src:
        dst = os.path.join(tmpd, os.path.basename(w)[:-4] if w.endswith('.ice') else os.path.basename(w))
        shutil.copyfile(w, dst)
        whl.append(dst)
    cmd = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile', '--disable-pip-version-check',
           '--upgrade', '--target', site] + whl
    log('installing wheels', wdir, '->', site)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError('pip install failed: ' + r.stdout[-2000:])
    shutil.rmtree(tmpd, ignore_errors=True)
    with open(mark, 'w') as f:
        f.write(want)


class _Anything:
    """Permissive placeholder returned by stub modules (import-time use only)."""
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Anything()
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        return _Anything()
    def __getitem__(self, k): return _Anything()
    def __setitem__(self, k, v): pass
    def __iter__(self): return iter(())
    def __bool__(self): return False


class _StubModule(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        v = type(k, (_Anything,), {}) if k[:1].isupper() else _Anything()
        setattr(self, k, v)
        return v


class _StubFinder:
    """Meta-path finder serving stub modules (and any submodule) for the given roots."""
    def __init__(self, roots):
        self.roots = set(roots)

    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] not in self.roots:
            return None
        import importlib.machinery as im
        return im.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = _StubModule(spec.name); m.__path__ = []; m.__ice_stub__ = True
        return m

    def exec_module(self, module):
        pass


# modules ms_pred imports at module level but never uses on the GLACIER predict path (pygmtools: sinkhorn / hungarian
# in the training losses only)
OPTIONAL = ('pygmtools', 'h5py', 'multiprocess', 'platformdirs')
SHIMMED = ('rdkit', 'dgl', 'torch_scatter', 'LinSATNet', 'ms_pred')


def setup_env(a, log):
    pkg = os.path.abspath(a.pkg)
    paths = [os.path.join(pkg, 'shim'), os.path.join(pkg, 'src')]
    if a.site.lower() != 'none':
        install_site(find_wheels(pkg, a.wheels), a.site, log)
        paths.append(os.path.abspath(a.site))
    for p in reversed(paths):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    for k in [k for k in sys.modules if k.split('.')[0] in SHIMMED]:
        del sys.modules[k]
    importlib.invalidate_caches()
    import warnings
    warnings.filterwarnings('ignore', category=SyntaxWarning)       # invalid escapes in ms_pred regex literals
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    # ms_pred modules that are training / plotting only: empty placeholders, so ray / matplotlib / seaborn /
    # rdkit.Chem.Draw are never imported (whether or not they are installed)
    for name in ('ms_pred.common.plot_utils', 'ms_pred.nn_utils.tune_utils', 'ms_pred.nn_utils.base_hyperopt'):
        mod = types.ModuleType(name); mod.__ice_placeholder__ = True
        sys.modules[name] = mod
    for f in ('plot_mol_as_vector', 'plot_compare_ms', 'plot_ms'):
        setattr(sys.modules['ms_pred.common.plot_utils'], f, lambda *x, **k: None)
    missing = []
    for m in OPTIONAL:
        try:
            importlib.import_module(m)
        except BaseException:
            for k in [k for k in sys.modules if k.split('.')[0] == m]:
                del sys.modules[k]
            missing.append(m)
    if missing:
        sys.meta_path.insert(0, _StubFinder(missing))
    log('stubbed (absent) modules:', missing)
    import rdkit
    if not rdkit.__version__.startswith(RDKIT_REQ) and not a.any_rdkit:
        raise RuntimeError(f'rdkit {rdkit.__version__} at {rdkit.__file__}; the GLACIER panel scores used {RDKIT_REQ}.x '
                           '-- pass --any-rdkit to override')
    return rdkit.__version__


# ------------------------------------------------------------------------------------------- spectral kernels ------
# verbatim copies of work/analysis/engine_v6/code/casmi/spectra.py (numba; pure-python fallback when numba is absent)
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                      # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

import numpy as np

LOG4 = np.log(4.0)
SIM_EPS = 1e-9


@njit(cache=False)
def clean_peaks(mz, it, prec_mz, floor, topk, max_over_prec):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if mz[i] <= prec_mz + max_over_prec and it[i] > mx:
            mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if mz[i] <= prec_mz + max_over_prec and it[i] >= thr and mz[i] > 0:
            c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if mz[i] <= prec_mz + max_over_prec and it[i] >= thr and mz[i] > 0:
            idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float64)
        for i in range(c):
            v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[o[i]]
        idx = k2
        c = topk
    om = np.empty(c, np.float64)
    oi = np.empty(c, np.float32)
    for i in range(c):
        om[i] = mz[idx[i]]
        oi[i] = it[idx[i]] / mx
    o = np.argsort(om)
    om2 = np.empty(c, np.float32)
    oi2 = np.empty(c, np.float32)
    for i in range(c):
        om2[i] = om[o[i]]
        oi2[i] = oi[o[i]]
    return om2, oi2


@njit(cache=False, fastmath=True)
def _weights(it, power, ent_weight):
    n = len(it)
    p = np.empty(n, np.float64)
    s = 0.0
    for i in range(n):
        v = it[i] ** power
        p[i] = v; s += v
    if s <= 0:
        return p
    for i in range(n):
        p[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(n):
            if p[i] > 0:
                S -= p[i] * np.log(p[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(n):
                p[i] = p[i] ** w; s2 += p[i]
            if s2 > 0:
                for i in range(n):
                    p[i] /= s2
    return p


@njit(cache=False, fastmath=True)
def prep_query(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    mx = 0.0
    for i in range(n):
        if it[i] > mx:
            mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float64), np.empty(0, np.float64)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr:
            c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float64)
        for i in range(c):
            v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float64)
    oi = np.empty(c, np.float64)
    for i in range(c):
        om[i] = mz[idx[i]]
        oi[i] = it[idx[i]]
    return om, _weights(oi, power, ent_weight)


@njit(cache=False, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol, ppm_tol):
    n = len(qmz); m = len(cmz)
    if n == 0 or m == 0:
        return 0.0
    SA = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    SAB = 0.0
    i = 0; j = 0
    while i < n and j < m:
        t = tol
        pt = qmz[i] * ppm_tol * 1e-6
        if pt > t:
            t = pt
        d = qmz[i] - cmz[j]
        if d < -t:
            v = qp[i] * 0.5; i += 1
        elif d > t:
            v = cp[j] * 0.5; j += 1
        else:
            v = (qp[i] + cp[j]) * 0.5; i += 1; j += 1
        if v > 0:
            SAB -= v * np.log(v)
    while i < n:
        v = qp[i] * 0.5
        if v > 0:
            SAB -= v * np.log(v)
        i += 1
    while j < m:
        v = cp[j] * 0.5
        if v > 0:
            SAB -= v * np.log(v)
        j += 1
    s = 1.0 - (2.0 * SAB - SA - SB) / LOG4
    if s < SIM_EPS:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return s


# ------------------------------------------------------------------------------------------ query conditions -------
def instr_family(s):
    """= casmi.build.instr_family (0 timsTOF, 1 Orbitrap/FT, 2 TOF, 3 IT, 4 QQQ, 5 other)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 5
    t = str(s).lower()
    if 'timstof' in t: return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t or 'ft-icr' in t or 'fticr' in t: return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qit' in t or 'lit' in t: return 3
    if 'qq' in t or 'quattro' in t or 'triple' in t: return 4
    return 5


def _nce(s):
    v = [float(x) for x in re.findall(r'[0-9]+\.?[0-9]*', str(s))]
    return float(np.mean(v)) if v else np.nan


def spectrum_ce(sp):
    """CE number as in iceberg/jobs.py: NCE units -> mean of collision_energy_orig numbers; else mean(ce_ev)."""
    if 'ce_units' in sp or 'ce_ev' in sp:
        if sp.get('ce_units') == 'NCE':
            return _nce(sp.get('ce_orig'))
        v = sp.get('ce_ev')
        return float(np.mean(v)) if v is not None and len(v) else np.nan
    v = sp.get('ce')
    return float(v) if v is not None and np.isfinite(float(v)) else np.nan


def ce_bucket(ce):
    ceb = float(np.round(ce / 5.0) * 5.0) if np.isfinite(ce) else np.nan
    if np.isfinite(ceb) and ceb < 5:
        ceb = 5.0
    return ceb


def is_covered(sp):
    md = sp.get('mode')
    pos = (md == 1) or (isinstance(md, str) and md.lower() in ('positive', 'pos', '+', '1'))
    return bool(pos and sp.get('adduct') in COV)


def instr_token(sp):
    if sp.get('instr') in ('Orbitrap', 'QTOF'):
        return sp['instr']
    return 'Orbitrap' if instr_family(sp.get('instrument')) == 1 else 'QTOF'


def prep_spectrum(sp):
    """raw peaks -> library-cache cleaning (float32) -> score.py cleaning -> prep_query."""
    mz = np.asarray(sp['mz'], np.float64); it = np.asarray(sp['it'], np.float64); prec = float(sp['prec'])
    m1, i1 = clean_peaks(mz, it, prec, 0.001, 512, 2.0)
    cm, ci = clean_peaks(m1.astype(np.float64), i1.astype(np.float64), prec, 0.001, 512, 2.0)
    return prep_query(cm.astype(np.float64), ci.astype(np.float64), FLOOR, TOPK, POWER, ENTW)


# --------------------------------------------------------------------------------------------- prediction ----------
def post(sp, k=100):
    """= a1_glacier/glacier_load.py::post: drop <= 0 intensity / mass -> merge identical m/z (4 dp) by MAX -> top-k
    -> sort by m/z -> max-normalise (float32)."""
    sp = sp[(sp[:, 1] > 0) & (sp[:, 0] > 0)]
    if len(sp) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    key = np.round(sp[:, 0].astype(np.float64), 4)
    u, inv = np.unique(key, return_inverse=True)
    it = np.zeros(len(u), np.float64)
    np.maximum.at(it, inv, sp[:, 1].astype(np.float64))
    if len(u) > k:
        o = np.argsort(-it, kind='stable')[:k]; u, it = u[o], it[o]
    o = np.argsort(u); u, it = u[o], it[o]
    return u.astype(np.float32), (it / it.max()).astype(np.float32)


def merge(specs):
    """= iceberg/score.py::merge = ice_ft/panel_score.py::merge (summed-CE prediction)."""
    mz = np.concatenate([s[0] for s in specs]); it = np.concatenate([s[1] / s[1].max() for s in specs])
    k = np.round(mz, 4); u, inv = np.unique(k, return_inverse=True)
    w = np.bincount(inv, weights=it)
    return u, w / w.max()


def batch_size_for(n_atoms, bs_max=32, budget=40 * 40 * 32):
    """= glacier_load.batch_size_for: rows of the reachability / edge_input tensors scale with B * N^2."""
    return int(max(1, min(bs_max, budget // max(1, n_atoms * n_atoms))))


def _align_onehot_tables(sd):
    """= glacier_load._align_onehot_tables: ms_pred.common's adduct / instrument one-hot tables vs checkpoint buffers."""
    import ms_pred.common as common
    notes = []
    if 'instrument_embedder' in sd:
        n_ck = sd['instrument_embedder'].shape[0]
        n_now = len(set(common.instrument2onehot_pos.values()))
        if n_ck != n_now:
            if n_ck == 2:
                for k in [k for k, v in common.instrument2onehot_pos.items() if v >= 2]:
                    del common.instrument2onehot_pos[k]
                notes.append('instrument table trimmed to {Orbitrap:0, QTOF:1} (2-slot checkpoint)')
            else:
                raise RuntimeError(f'instrument_embedder has {n_ck} slots, ms_pred.common has {n_now}')
    if 'adduct_embedder' in sd:
        n_ck = sd['adduct_embedder'].shape[0]
        n_now = len(set(common.ion2onehot_pos.values()))
        if n_ck != n_now:
            raise RuntimeError(f'adduct_embedder has {n_ck} rows, ms_pred.common has {n_now} adduct types')
    return notes


def load_model(pkg, dev, log, ckpt=None):
    import torch, tempfile
    import ms_pred.common.chem_utils as cu
    # chem_utils imports platformdirs.user_cache_dir at module level (tautomer-transform file of the InChI round trip;
    # not on the GLACIER path) -> point it at a writable temp dir as ice_runner does (platformdirs itself not needed)
    cache_root = os.path.join(tempfile.gettempdir(), 'ice_ms_pred_cache')
    cu.user_cache_dir = lambda name, *x, **k: os.path.join(cache_root, name)
    from ms_pred.glacier import joint_model as gl_jm
    c = torch.load(ckpt or os.path.join(pkg, 'ckpt', 'glacier.pt'), map_location='cpu', weights_only=True)
    sd = c['state_dict']
    hp = {k: v for k, v in dict(c['hyper_parameters']).items() if not str(k).startswith('_')}
    notes = _align_onehot_tables(sd)
    m = gl_jm.JointModel(**hp)
    m.load_state_dict(sd, strict=True)
    m = m.to(dev).eval()
    m.freeze()
    m._predict_prepared = True
    log('GLACIER loaded:', f'{sum(p.numel() for p in m.parameters()):,} params', notes or '')
    return m


def canon(smi):
    """= glacier_load.canon (predict_smis_joint.prepare_entry canonicalisation); None = rejected (same rules)."""
    from rdkit import Chem
    from ms_pred.common import chem_utils
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in chem_utils.VALID_ATOM_NUM:
                return None
        mol = Chem.RemoveHs(mol)
        if mol.GetNumAtoms() > 100:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


_CANON = {}          # smiles -> (canonical smiles | None, heavy atoms, reason)


def smiles_ok(smi):
    """Cheap pre-check (the canonicalisation above) so one bad SMILES does not fail a whole batch. -> (ok, reason)"""
    if smi in _CANON:
        c = _CANON[smi]
        return c[0] is not None, c[2]
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        if not isinstance(smi, str) or not smi:
            _CANON[smi] = (None, 0, 'empty'); return False, 'empty'
        c = canon(smi)
        if c is None:
            m = Chem.MolFromSmiles(smi)
            why = 'rdkit parse' if m is None else ('> 100 heavy atoms' if Chem.RemoveHs(m).GetNumAtoms() > 100
                                                   else 'element outside VALID_ATOM_NUM')
            _CANON[smi] = (None, 0, why); return False, why
        mc = Chem.MolFromSmiles(c)
        if mc is None or mc.GetNumAtoms() == 0:
            _CANON[smi] = (None, 0, 'canonical re-parse'); return False, 'canonical re-parse'
        _CANON[smi] = (c, int(mc.GetNumHeavyAtoms()), ''); return True, ''
    except Exception as e:
        why = f'exception {e!r}'[:200]
        _CANON[smi] = (None, 0, why)
        return False, why


class Featurizer:
    """= glacier_load.Featurizer: GraphormerFeatDataset._featurize_cached equivalent (LRU per (canonical, adduct))."""

    def __init__(self, model, cache=512):
        from ms_pred.glacier import dataset as gl_ds
        import ms_pred.common as common
        self.common = common
        self.tp = gl_ds.TreeProcessor(pe_embed_k=model.pe_embed_k, root_encode='graphormer',
                                      embed_elem_group=model.embed_elem_group,
                                      multi_hop_max_dist=model.multi_hop_max_dist)
        self.cache = OrderedDict(); self.cap = cache

    def item(self, smi_c, adduct, ce, instr, name=''):
        common = self.common
        key = (smi_c, adduct)
        feat = self.cache.get(key)
        if feat is None:
            tree = {'root_canonical_smiles': smi_c, 'adduct': adduct, 'collision_energy': ce}
            feat = self.tp.featurize_tree(tree, include_inten_targs=False, include_frag_targs=False)
            self.cache[key] = feat
            if len(self.cache) > self.cap:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        feat = dict(feat)
        feat['collision_energy'] = float(ce)
        it = {'name': name, 'adduct': common.ion2onehot_pos[adduct], 'precursor': 0.0,
              'instrument': common.instrument2onehot_pos.get(instr, common.instrument2onehot_pos.get('Orbitrap', 0))}
        it.update(feat)
        return it


def _batch_to_device(batch, device):
    """= ms_pred.glacier.predict_smis_joint._batch_to_device."""
    import torch
    for k in batch:
        v = batch[k]
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
        elif k == 'graphormer_input' and v is not None:
            for gk in v:
                v[gk] = v[gk].to(device, non_blocking=True)
    return batch


class Predictor:
    def __init__(self, model, dev, bs, log):
        from ms_pred.glacier import dataset as gl_ds
        self.m, self.dev, self.bs, self.log = model, dev, bs, log
        self.collate = gl_ds.IntenDataset.collate_fn
        self.fz = Featurizer(model)
        self.cache = {}        # (smiles, adduct, instr, ce) -> (mz f64, it f64) | None
        self.pcache = {}       # (canonical, adduct, instr, ce) -> same (identical canonical forms predicted once)
        self.n_pred = 0; self.n_fail = 0; self.n_feat_fail = 0; self.t_pred = 0.0; self.last_err = ''

    def _run(self, keys):
        outs = [None] * len(keys)
        items, idx = [], []
        for n, (c, ad, ins, ce) in enumerate(keys):
            try:
                items.append(self.fz.item(c, ad, float(ce), ins)); idx.append(n)
            except Exception as e:                       # featurisation failure -> that job only
                self.n_feat_fail += 1
                if self.n_feat_fail <= 5:
                    self.log('featurise fail', c[:80], ad, repr(e)[:160])
        if items:
            batch = _batch_to_device(self.collate(items), self.dev)
            raw = self.m.predict_inten_frag_batch(batch)['spec']
            for n, s in zip(idx, raw):
                outs[n] = post(s.float().cpu().numpy())
        return outs

    def _safe(self, keys):
        import torch
        try:
            return self._run(keys)
        except Exception as e:
            self.last_err = repr(e)[:300]
            self.log('batch fail -> split', len(keys), repr(e)[:160])
            if self.dev != 'cpu':
                torch.cuda.empty_cache()
            if 'out of memory' in str(e).lower() and len(keys) > 4:     # GPU tight: smaller batches from now on
                nb = max(4, len(keys) // 2)                              # half of the size that FAILED (ice_runner)
                if nb < self.bs:
                    self.log(f'OOM -> max batch {self.bs} -> {nb}')
                    self.bs = nb
        if len(keys) == 1:
            return [None]
        out = []
        cs = max(1, len(keys) // 4)
        for c0 in range(0, len(keys), cs):
            out.extend(self._safe(keys[c0:c0 + cs]))
        return out

    def predict(self, jobs, deadline):
        """jobs: set of (smiles, adduct, instr, ce). Fills self.cache. Returns False if the deadline stopped it."""
        todo = sorted(j for j in jobs if j not in self.cache)
        need = {}
        for smi, ad, ins, ce in todo:
            c = _CANON.get(smi, (None,))[0]
            if c is None:                                   # never reached for pre-checked candidates
                self.cache[(smi, ad, ins, ce)] = None
                continue
            k = (c, ad, ins, float(ce))
            if k in self.pcache:
                self.cache[(smi, ad, ins, ce)] = self.pcache[k]
                continue
            need.setdefault(k, []).append((smi, ad, ins, ce))
        heavy = {k: _CANON[v[0][0]][1] for k, v in need.items()}
        keys = sorted(need, key=lambda k: (heavy[k], k[0], k[1], k[2], k[3]))
        t0 = time.time()
        i = 0
        while i < len(keys):
            if time.time() > deadline:
                self.t_pred += time.time() - t0
                return False
            L = max(heavy[k] for k in keys[i:i + self.bs])
            bs = batch_size_for(L, self.bs, self.bs * ATOM_BUDGET_PER_BS)
            sub = keys[i:i + bs]; i += len(sub)
            outs = self._safe(sub)
            for k, o in zip(sub, outs):
                ok = o is not None and len(o[0]) > 0
                v = (o[0].astype(np.float64), o[1].astype(np.float64)) if ok else None
                self.pcache[k] = v
                for j in need[k]:
                    self.cache[j] = v
                self.n_pred += 1; self.n_fail += int(o is None)
            if self.n_pred >= FAILFAST_N and self.n_fail == self.n_pred:
                # systematic failure (e.g. an untested device path): stop now instead of splitting every batch down
                # to single jobs for the whole budget -> status error, all null (= notebook falls back to ICE only)
                raise RuntimeError(f'first {self.n_pred} predictions all failed -> abort; last error: {self.last_err}')
        self.t_pred += time.time() - t0
        return True


# ------------------------------------------------------------------------------------------------ scoring ----------
def molecule_plan(mol):
    """Covered spectra -> list of (qm, qp, adduct, instr, ce_bucket)."""
    out = []
    for sp in mol.get('spectra') or []:
        if not is_covered(sp):
            continue
        qm, qp = prep_spectrum(sp)
        out.append((qm, qp, sp['adduct'], instr_token(sp), ce_bucket(spectrum_ce(sp))))
    return out


def molecule_jobs(plan, cands):
    J = set()
    for _, _, ad, ins, ceb in plan:
        for c in cands:
            if np.isfinite(ceb):
                J.add((c, ad, ins, ceb))
            else:
                for ce in SUM_CE:
                    J.add((c, ad, ins, ce))
    return J


def score_molecule(plan, cands, cache):
    res = {}
    sum_cache = {}
    for c in cands:
        sims = []
        for qm, qp, ad, ins, ceb in plan:
            if np.isfinite(ceb):
                pr = cache.get((c, ad, ins, ceb))
            else:
                sk = (c, ad, ins)
                if sk not in sum_cache:
                    parts = [cache.get((c, ad, ins, ce)) for ce in SUM_CE]
                    parts = [p for p in parts if p is not None]
                    sum_cache[sk] = merge(parts) if len(parts) == len(SUM_CE) else None
                pr = sum_cache[sk]
            if pr is None:
                continue
            pm, pp = prep_query(pr[0], pr[1], FLOOR, TOPK, POWER, ENTW)
            s = entropy_sim(qm, qp, pm, pp, TOL, PPM)
            if np.isfinite(s):          # = panel np.nanmean (NaN sims skipped); also keeps out.json NaN-free
                sims.append(s)
        res[c] = float(np.mean(sims)) if sims else None
    return res


# ---------------------------------------------------------------------------------------------------- main ---------
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('pkg'); ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--device', default='cuda', help='cuda | cpu (cuda falls back to cpu when unavailable)')
    ap.add_argument('--budget', type=float, default=0.0, help='wall-clock budget in s for the whole run (0 = none)')
    ap.add_argument('--site', default=os.environ.get('ICE_SITE', '/tmp/ice_site'),
                    help="dir of the offline wheel install (the ICE runner's; filled here if missing); 'none' = current env")
    ap.add_argument('--wheels', default=None, help='dir with the casmi26-iceberg *.whl.ice (used only if --site is not ready)')
    ap.add_argument('--bs', type=int, default=32, help='max prediction batch (auto-reduced for large molecules: B*N^2 <= bs*1600)')
    ap.add_argument('--chunk-jobs', type=int, default=3000, help='predictions per chunk (checkpoint granularity)')
    ap.add_argument('--threads', type=int, default=0, help='torch CPU threads (0 = torch default)')
    ap.add_argument('--memfrac', type=float, default=0.0, help='torch.cuda per-process memory fraction (0 = none)')
    ap.add_argument('--nondeterministic', dest='deterministic', action='store_false',
                    help='do not enable torch deterministic algorithms')
    ap.add_argument('--any-rdkit', action='store_true', help='allow an RDKit other than 2025.03 (NOT panel-exact)')
    ap.add_argument('--ckpt', default=None, help='checkpoint file (default <pkg>/ckpt/glacier.pt)')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.deterministic:                                          # before torch / CUDA initialise
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    log = Log(a.quiet)
    t_start = time.time()
    deadline = t_start + a.budget if a.budget > 0 else float('inf')
    meta = dict(model='glacier', status='init', n_mols=0, n_mols_scored=0, n_mols_covered=0, n_cands=0, n_scored=0,
                errors=[])
    try:
        mols = json.load(open(a.inp))
    except Exception as e:
        meta.update(status='bad_input', errors=[repr(e)[:500]])
        write_json(a.out, {}); write_json(a.out + '.meta.json', meta)
        return 0
    result = {}
    for mol in mols:
        result[str(mol.get('mid'))] = {c: None for c in (mol.get('cands') or [])}
    meta.update(n_mols=len(result), n_cands=sum(len(v) for v in result.values()))
    write_json(a.out, result)                                    # valid (all-null) output from the start

    def done(status):
        meta.update(status=status, seconds=round(time.time() - t_start, 2), log_tail=log.lines[-40:])
        write_json(a.out, result); write_json(a.out + '.meta.json', meta)
        log('status', status, {k: meta[k] for k in ('n_mols', 'n_mols_covered', 'n_mols_scored', 'n_cands', 'n_scored')})
        return 0

    try:
        meta['rdkit'] = setup_env(a, log)
        import torch
        meta['torch'] = torch.__version__
        dev = a.device
        if dev.startswith('cuda') and not torch.cuda.is_available():
            log('cuda unavailable -> cpu'); dev = 'cpu'
        meta['device'] = dev
        if a.threads > 0:
            torch.set_num_threads(a.threads)
        if dev.startswith('cuda') and a.memfrac > 0:
            torch.cuda.set_per_process_memory_fraction(a.memfrac)
        if a.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        pkg = os.path.abspath(a.pkg)
        model = load_model(pkg, dev, log, a.ckpt)
        meta['t_load'] = round(time.time() - t_start, 2)
        log(f'model loaded on {dev} (rdkit {meta["rdkit"]}, torch {torch.__version__}, numba {HAVE_NUMBA}, '
            f'threads {torch.get_num_threads()})')
        pred = Predictor(model, dev, a.bs, log)
        # plan every molecule (cheap), then predict chunk by chunk in input order
        plans, bad = {}, {}
        for mol in mols:
            mid = str(mol.get('mid'))
            try:
                plans[mid] = molecule_plan(mol)
            except Exception as e:
                meta['errors'].append(f'{mid}: plan {repr(e)[:200]}'); plans[mid] = []
        meta['n_mols_covered'] = sum(1 for p in plans.values() if p)
        okc = {}
        chunk, chunk_jobs = [], set()

        def flush_chunk():
            nonlocal chunk, chunk_jobs
            if not chunk:
                return True
            t0 = time.time()
            finished = pred.predict(chunk_jobs, deadline)
            if finished:
                for mid, cands in chunk:
                    try:
                        sc = score_molecule(plans[mid], cands, pred.cache)
                        result[mid].update(sc)
                        meta['n_mols_scored'] += 1
                        meta['n_scored'] += sum(v is not None for v in sc.values())
                    except Exception as e:
                        meta['errors'].append(f'{mid}: score {repr(e)[:200]}')
                write_json(a.out, result)
                log(f'chunk: {len(chunk)} mols {len(chunk_jobs)} jobs {time.time() - t0:.1f}s '
                    f'({meta["n_mols_scored"]}/{meta["n_mols_covered"]} covered mols scored)')
            chunk, chunk_jobs = [], set()
            return finished

        stopped = False
        for mol in mols:
            mid = str(mol.get('mid'))
            if not plans.get(mid):
                continue
            cands = list(dict.fromkeys(mol.get('cands') or []))
            for c in cands:
                if c not in okc:
                    okc[c], why = smiles_ok(c)
                    if not okc[c]:
                        nbad = sum(1 for v in okc.values() if not v)
                        if nbad <= 5:
                            meta['errors'].append(f'smiles rejected ({why}): {str(c)[:100]}')
                        if nbad >= 50 and nbad == len(okc):                  # environment problem, not the data
                            raise RuntimeError(f'first {nbad} SMILES all rejected, last: {why}')
            cands = [c for c in cands if okc[c]]
            if not cands:
                continue
            chunk.append((mid, cands))
            chunk_jobs |= molecule_jobs(plans[mid], cands)
            if len(chunk_jobs) >= a.chunk_jobs:
                if not flush_chunk():
                    stopped = True
                    break
        if not stopped and not flush_chunk():
            stopped = True
        meta.update(n_pred=pred.n_pred, n_pred_fail=pred.n_fail, n_feat_fail=pred.n_feat_fail,
                    t_pred=round(pred.t_pred, 2), n_bad_smiles=sum(1 for v in okc.values() if not v),
                    pred_per_s=round(pred.n_pred / max(pred.t_pred, 1e-9), 2), bs_final=pred.bs)
        if dev.startswith('cuda'):
            meta['max_mem_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        return done('budget' if stopped else 'ok')
    except BaseException as e:                                  # never crash the notebook
        meta['errors'].append(traceback.format_exc()[-3000:])
        log('ERROR', repr(e)[:300])
        try:
            return done('error')
        except Exception:
            return 2


if __name__ == '__main__':
    sys.exit(main())
