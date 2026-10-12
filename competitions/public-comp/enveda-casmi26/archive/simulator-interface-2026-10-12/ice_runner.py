#!/usr/bin/env python
"""ICEBERG isomer re-scorer -- self-contained subprocess runner for the CASMI26 Kaggle notebook.

    python ice_runner.py <package_dir> <in.json> <out.json> [--device cuda] [--budget SEC] [--site DIR|none] ...

Runs in its OWN process with its own sys.path (package shim + ms_pred src + an offline --target install of the shipped
wheels incl. RDKit 2025.03), so the notebook process keeps RDKit 2026.03 and never imports torch-DGL stand-ins.

in.json  : [{"mid": str, "cands": [smiles, ...],
             "spectra": [{"mz": [...], "it": [...], "prec": float, "adduct": "[M+H]+", "mode": "positive" | +1 | -1,
                          "instrument": raw instrument_type string (optional; default -> QTOF token),
                          "ce_ev": [...], "ce_orig": str, "ce_units": "eV" | "NCE"     (raw test-parquet fields), or
                          "ce": float | null   (already-resolved CE, used only when ce_ev / ce_units are absent)}]}]
out.json : {mid: {smiles: ice_ex_mean or null}}   (null = no covered spectrum / prediction failed / not reached)
out.json.meta.json : run statistics (status, timings, versions, errors).

Score = the iceberg-track panel definition `ice_ex_mean` (work/analysis/fwd/iceberg/{jobs,predict_run,score}.py):
  covered spectrum : mode +1 and adduct in {[M+H]+, [M+Na]+}
  instrument token : 'Orbitrap' if instr_family(instrument_type) == 1 else 'QTOF'
  CE               : units == 'NCE' -> mean of the numbers in collision_energy_orig, else mean(collision_energy_ev)
                     (NaN if empty); bucket = np.round(ce / 5) * 5, min 5; NaN -> 'sum' prediction (CE 20+40+60 merged)
  prediction       : JointModel.predict_mol(threshold 0, max_nodes 100, precursor_mz 0) -> drop 0 intensity -> merge
                     identical m/z (round 4) -> top 100 -> max 1 (float32)
  query            : raw peaks -> clean_peaks(prec, floor 0.001, top 512, <= prec + 2) (library cache, float32) ->
                     clean_peaks again (score.py) -> prep_query(floor 0.002, top 256, power 1, entropy weighting)
  similarity       : engine entropy_sim, 0.01 Da / 20 ppm;   ice_ex_mean = mean over the molecule's covered spectra
                     that have a prediction (NaN sims skipped).
Robustness: the output file is written (all null) before anything heavy happens and rewritten after every chunk, so a
crash, a kill or the time budget always leaves a valid out.json; the process exits 0 whenever out.json was written.
"""
import os, sys, json, time, re, math, argparse, traceback, subprocess, glob, types, importlib
sys.dont_write_bytecode = True          # never write __pycache__ into the (read-only on Kaggle) package dir

COV = ('[M+H]+', '[M+Na]+')
SUM_CE = (20.0, 40.0, 60.0)
FLOOR, TOPK, POWER, ENTW, TOL, PPM = 0.002, 256, 1.0, True, 0.01, 20.0
RDKIT_REQ = '2025.03'


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
        s = f'[ice {time.time() - self.t0:7.1f}s] ' + ' '.join(str(x) for x in a)
        self.lines.append(s)
        if not self.quiet:
            print(s, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------- environment -----
def install_site(pkg, site, log):
    """Offline install of pkg/wheels/*.whl.ice (= *.whl renamed so that the notebook's own `rdkit-*.whl` glob can
    never pick them up) into `site` with pip --no-index --no-deps --target; idempotent via a marker file."""
    import shutil
    src = sorted(glob.glob(os.path.join(pkg, 'wheels', '*.whl.ice')) + glob.glob(os.path.join(pkg, 'wheels', '*.whl')))
    mark = os.path.join(site, '.ice_site_ok')
    want = '\n'.join(os.path.basename(w) for w in src)
    if os.path.exists(mark) and open(mark).read() == want:
        return
    if not src:
        raise RuntimeError(f'no wheels in {pkg}/wheels')
    wdir = os.path.join(site, '.wheels')
    os.makedirs(wdir, exist_ok=True)
    whl = []
    for w in src:
        dst = os.path.join(wdir, os.path.basename(w)[:-4] if w.endswith('.ice') else os.path.basename(w))
        shutil.copyfile(w, dst)
        whl.append(dst)
    cmd = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile', '--disable-pip-version-check',
           '--upgrade', '--target', site] + whl
    log('installing wheels ->', site)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError('pip install failed: ' + r.stdout[-2000:])
    shutil.rmtree(wdir, ignore_errors=True)
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


# modules ms_pred imports at module level but never uses on the ICEBERG predict_mol path
OPTIONAL = ('pygmtools', 'h5py', 'multiprocess', 'platformdirs')


def setup_env(a, log):
    pkg = os.path.abspath(a.pkg)
    paths = [os.path.join(pkg, 'shim'), os.path.join(pkg, 'src')]
    if a.site.lower() != 'none':
        install_site(pkg, a.site, log)
        paths.append(os.path.abspath(a.site))
    for p in reversed(paths):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    for k in [k for k in sys.modules if k.split('.')[0] in ('rdkit', 'dgl', 'torch_scatter', 'ms_pred')]:
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
        raise RuntimeError(f'rdkit {rdkit.__version__} at {rdkit.__file__}; ICEBERG needs {RDKIT_REQ}.x '
                           '(Kekulize differs in 2026.03) -- pass --any-rdkit to override')
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
    """= iceberg/predict_run.py::post."""
    sp = sp[sp[:, 1] > 0]
    if len(sp) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    key = np.round(sp[:, 0].astype(np.float64), 4)
    u, inv = np.unique(key, return_inverse=True)
    it = np.bincount(inv, weights=sp[:, 1].astype(np.float64))
    if len(u) > k:
        o = np.argsort(-it)[:k]; u, it = u[o], it[o]
    o = np.argsort(u); u, it = u[o], it[o]
    return u.astype(np.float32), (it / it.max()).astype(np.float32)


def merge(specs):
    """= iceberg/score.py::merge (summed-CE prediction)."""
    mz = np.concatenate([s[0] for s in specs]); it = np.concatenate([s[1] / s[1].max() for s in specs])
    k = np.round(mz, 4); u, inv = np.unique(k, return_inverse=True)
    w = np.bincount(inv, weights=it)
    return u, w / w.max()


def load_model(pkg, dev, log):
    import torch, tempfile
    import ms_pred.common.chem_utils as cu
    # canonical_mol_from_inchi (used by predict_mol's InChI round trip) writes a fixed tautomer-transform file to
    # platformdirs.user_cache_dir; point it at a writable temp dir (content is constant -> results unchanged, and
    # platformdirs itself is not needed)
    cache_root = os.path.join(tempfile.gettempdir(), 'ice_ms_pred_cache')
    cu.user_cache_dir = lambda name, *x, **k: os.path.join(cache_root, name)
    log('ms_pred tautomer mode', cu._RD_TAUTOMER_CANONICALIZER, 'cache', cache_root)
    from ms_pred.iceberg import gen_model, inten_model, joint_model
    mods = []
    for cls, f in ((gen_model.FragGNN, 'gen.pt'), (inten_model.IntenGNN, 'inten.pt')):
        c = torch.load(os.path.join(pkg, 'ckpt', f), map_location='cpu', weights_only=True)
        m = cls(**c['hyper_parameters'])
        m.load_state_dict(c['state_dict'], strict=True)
        mods.append(m)
    jm = joint_model.JointModel(mods[0], mods[1]).to(dev)
    jm.eval()
    _speed_patches()
    return jm


def _speed_patches():
    """= iceberg/ice.py speed patches (same math; used for every panel prediction)."""
    import torch
    import ms_pred.nn_utils as _nnu
    import ms_pred.nn_utils.nn_utils as _nnu_mod
    import ms_pred.nn_utils.dgl_modules as _dglm

    def _pack_padded_tensor(input, lengths):
        lengths = torch.as_tensor(lengths, device=input.device).long()
        mask = torch.arange(input.shape[1], device=input.device)[None, :] < lengths[:, None]
        return input[mask]

    def _ggc_forward(self, graph, feat, etypes=None):
        zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
        feat = torch.cat([feat, zero_pad], -1)
        src, dst = graph.edges()
        src = src.long(); dst = dst.long()
        groups = [(i, torch.nonzero(etypes == i, as_tuple=True)[0]) for i in range(self._n_etypes)]
        groups = [(i, e) for i, e in groups if e.numel() > 0]
        for _ in range(self._n_steps):
            m = feat.new_zeros((src.numel(), self._out_feats))
            for i, e in groups:
                m[e] = self.linears[i](feat[src[e]])
            a = feat.new_zeros((feat.shape[0], self._out_feats)).index_add_(0, dst, m)
            feat = self.gru(a, feat)
        return feat

    _nnu.pack_padded_tensor = _pack_padded_tensor
    _nnu_mod.pack_padded_tensor = _pack_padded_tensor
    _dglm.GatedGraphConv.forward = _ggc_forward


def smiles_ok(smi):
    """Cheap pre-check so one bad SMILES does not fail (and re-run) a whole batch. -> (ok, reason)"""
    try:
        import ms_pred.common as common
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumAtoms() == 0:
            return False, 'rdkit parse'
        els = set(common.VALID_ELEMENTS)
        bad = sorted({at.GetSymbol() for at in m.GetAtoms()} - els)
        if bad:
            return False, f'elements {bad}'
        r = common.smiles_from_inchi(common.inchi_from_smiles(common.rm_stereo(smi)))
        if not r or Chem.MolFromSmiles(r) is None:
            return False, 'inchi round trip'
        return True, ''
    except Exception as e:
        return False, f'exception {e!r}'[:200]


class Predictor:
    def __init__(self, model, dev, bs, log):
        self.m, self.dev, self.bs, self.log = model, dev, bs, log
        self.cache = {}        # (smiles, adduct, instr, ce) -> (mz f64, it f64) | None
        self.n_pred = 0; self.n_fail = 0; self.t_pred = 0.0

    def _run(self, smis, adduct, instr, ce):
        import torch
        nb = len(smis)
        with torch.no_grad():
            r = self.m.predict_mol(smis, collision_eng=[float(ce)] * nb, precursor_mz=[0.0] * nb, adduct=[adduct] * nb,
                                   threshold=0.0, device=self.dev, max_nodes=100, instrument=[instr] * nb,
                                   binned_out=False)
        return [post(s.float().cpu().numpy()) for s in r['spec']]

    def _safe(self, smis, cond):
        import torch
        try:
            return self._run(smis, *cond)
        except Exception as e:
            self.log('batch fail -> split', cond, len(smis), repr(e)[:160])
            if self.dev != 'cpu':
                torch.cuda.empty_cache()
            if 'out of memory' in str(e).lower() and len(smis) > 4:     # GPU tight: smaller batches from now on
                # half of the size that FAILED (not of min(bs, len)): the quarter-splits of one failed batch used to
                # halve bs again on every OOM at the same size (96 -> 48 -> 12 -> 6 -> 4 within one batch)
                nb = max(4, len(smis) // 2)
                if nb < self.bs:
                    self.log(f'OOM -> max batch {self.bs} -> {nb}')
                    self.bs = nb
        if len(smis) == 1:
            return [None]
        out = []
        cs = max(1, len(smis) // 4)
        for c0 in range(0, len(smis), cs):
            out.extend(self._safe(smis[c0:c0 + cs], cond))
        return out

    def predict(self, jobs, deadline):
        """jobs: set of (smiles, adduct, instr, ce). Fills self.cache. Returns False if the deadline stopped it."""
        todo = sorted(j for j in jobs if j not in self.cache)
        conds = {}
        for smi, ad, ins, ce in todo:
            conds.setdefault((ad, ins, ce), []).append(smi)
        t0 = time.time()
        for cond in sorted(conds):
            smis = sorted(conds[cond], key=lambda s: (len(s), s))       # = predict_run order: natoms = len(SMILES)
            i = 0
            while i < len(smis):
                if time.time() > deadline:
                    self.t_pred += time.time() - t0
                    return False
                L = len(smis[min(i + self.bs, len(smis)) - 1])
                bs = self.bs if L <= 60 else (self.bs // 2 if L <= 90 else (self.bs // 4 if L <= 140 else self.bs // 8))
                sub = smis[i:i + max(1, bs)]; i += len(sub)
                outs = self._safe(sub, cond)
                for s, o in zip(sub, outs):
                    ok = o is not None and len(o[0]) > 0
                    self.cache[(s,) + cond] = (o[0].astype(np.float64), o[1].astype(np.float64)) if ok else None
                    self.n_pred += 1; self.n_fail += int(o is None)
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
                sims.append(s)          # (write_json uses allow_nan=False: one NaN used to abort the whole run)
        res[c] = float(np.mean(sims)) if sims else None
    return res


# ---------------------------------------------------------------------------------------------------- main ---------
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('pkg'); ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--device', default='cuda', help='cuda | cpu (cuda falls back to cpu when unavailable)')
    ap.add_argument('--budget', type=float, default=0.0, help='wall-clock budget in s for the whole run (0 = none)')
    ap.add_argument('--site', default=os.environ.get('ICE_SITE', '/tmp/ice_site'),
                    help="dir for the offline wheel install (created on first use); 'none' = use current env")
    ap.add_argument('--bs', type=int, default=96, help='max prediction batch (auto-reduced for long SMILES)')
    ap.add_argument('--chunk-jobs', type=int, default=3000, help='predictions per chunk (checkpoint granularity)')
    ap.add_argument('--threads', type=int, default=0, help='torch CPU threads (0 = torch default)')
    ap.add_argument('--memfrac', type=float, default=0.0, help='torch.cuda per-process memory fraction (0 = none)')
    ap.add_argument('--nondeterministic', dest='deterministic', action='store_false',
                    help='do not enable torch deterministic algorithms')
    ap.add_argument('--any-rdkit', action='store_true', help='allow an RDKit other than 2025.03 (NOT panel-exact)')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.deterministic:                                          # before torch / CUDA initialise
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    log = Log(a.quiet)
    t_start = time.time()
    deadline = t_start + a.budget if a.budget > 0 else float('inf')
    meta = dict(status='init', n_mols=0, n_mols_scored=0, n_mols_covered=0, n_cands=0, n_scored=0, errors=[])
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
        model = load_model(pkg, dev, log)
        meta['t_load'] = round(time.time() - t_start, 2)
        log(f'model loaded on {dev} (rdkit {meta["rdkit"]}, torch {torch.__version__}, numba {HAVE_NUMBA})')
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
                            meta['errors'].append(f'smiles rejected ({why}): {c[:100]}')
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
        meta.update(n_pred=pred.n_pred, n_pred_fail=pred.n_fail, t_pred=round(pred.t_pred, 2),
                    n_bad_smiles=sum(1 for v in okc.values() if not v),
                    pred_per_s=round(pred.n_pred / max(pred.t_pred, 1e-9), 2))
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
