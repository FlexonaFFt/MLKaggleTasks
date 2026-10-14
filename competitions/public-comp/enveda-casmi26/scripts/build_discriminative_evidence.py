'''Independent evidence experiments; retain credited Initial01 numerical kernels.'''
import copy
import json
from build_evidence_gap_iterations import COMMON, COVERAGE
from build_evidence_routing_research import build as initial_build
from build_flow_experiments import ROOT, prose, validate
from build_metric_safe_adaptation import code_cell

CONSENSUS = '''
class SimulatorConflictCheck(FusionFirstCoverage):
    def prepare(self, control):
        self.audit, self.plans = [], {}
        self.targets = {'gl': {}, 'ice': {}}
        scope = self.scope
        frames = {str(mid): frame for mid, frame in scope['te'].groupby('molecule_id', sort=False)}
        for row in control.itertuples(index=False):
            mid = row.molecule_id
            if mid not in frames:
                continue
            frame = frames[mid]
            if not ((frame.ionization_mode == 'positive') & frame.adduct.isin(['[M+H]+','[M+Na]+'])).any():
                continue
            c, s, k, f = self.fused(mid)
            if not c:
                continue
            order = self.baseline_order(mid, c, s, k, f)
            current = [c[i] for i in order]
            assert scope['safe_extend'](current[:25], current[25:] + row.smiles.split(';')) == row.smiles.split(';')
            gl = scope['gl_of'](mid)
            groups = {}
            for i, formula in enumerate(f):
                groups.setdefault(formula, []).append(i)
            plans = []
            for formula, indices in groups.items():
                if not 2 <= len(indices) <= 8:
                    continue
                values = [self.value(gl, c[i], k[i]) for i in indices]
                winner = self.unique_winner(values)
                if winner is None:
                    continue
                challenger = indices[winner]
                baseline_leader = next(i for i in order if i in indices)
                if challenger == indices[0] or challenger == baseline_leader:
                    continue
                plans.append((formula, indices, challenger))
                self.audit.append(dict(molecule_id=mid, formula=formula, candidates=len(indices), accepted=False))
                ice = scope['ICE_SCORES'].get(mid, {})
                if not all(self.finite(self.value(ice,c[i],k[i])) for i in indices):
                    self.targets['ice'].setdefault(mid, []).append(indices)
            if plans:
                self.plans[mid] = (c,s,k,f,order,plans)

    def unique_winner(self, values):
        if not values or not all(self.finite(v) for v in values):
            return None
        ordered = sorted(range(len(values)), key=lambda i: float(values[i]), reverse=True)
        if float(values[ordered[0]]) <= 0 or float(values[ordered[0]]) - float(values[ordered[1]]) <= 1e-8:
            return None
        return ordered[0]

    def run(self):
        control = self.control()
        self.prepare(control)
        predicted = self.predict()['ice']
        rows, accepted = [], 0
        for row in control.itertuples(index=False):
            mid, original = row.molecule_id, row.smiles.split(';')
            revised = list(original)
            if mid in self.plans:
                c,s,k,f,baseline,plans = self.plans[mid]
                current = list(baseline)
                for formula, indices, challenger in plans:
                    # Use one scoring pass for an entire group; never mix batch-normalized scores.
                    old = self.scope['ICE_SCORES'].get(mid,{})
                    evidence = old if all(self.finite(self.value(old,c[i],k[i])) for i in indices) else predicted.get(mid,{})
                    winner = self.unique_winner([self.value(evidence,c[i],k[i]) for i in indices])
                    if winner is None or indices[winner] != challenger:
                        continue
                    slots = [j for j,i in enumerate(current) if i in indices]
                    position = current.index(challenger)
                    current[slots[0]], current[position] = current[position], current[slots[0]]
                    accepted += 1
                    next(a for a in self.audit if a['molecule_id']==mid and a['formula']==formula)['accepted'] = True
                eligible = {j for j,i in enumerate(baseline) if any(i in p[1] for p in plans)}
                assert sorted(current) == sorted(baseline)
                assert all(current[j] == baseline[j] for j in range(len(current)) if j not in eligible)
                ordered = [c[i] for i in current]
                revised = self.scope['safe_extend'](ordered[:25], ordered[25:]+original)
            rows.append((mid,';'.join(revised)))
        return self.finish(rows,control,'simulator_conflict_check',dict(target_groups=len(self.audit),
            accepted_groups=accepted, max_group_size=8, ice_budget_seconds=1800 if self.scope['IS_RERUN'] else 120))
'''

DISCRIMINATIVE = '''
class DiscriminativeFragmentFallback(EvidenceIteration):
    @staticmethod
    def contrast(credits, intensity):
        # A shared explanation carries no information about which candidate is correct.
        support = (credits > 0).sum(axis=0)
        n = credits.shape[0]
        information = np.log(n / np.maximum(support,1)) / np.log(n)
        informative = (support > 0) & (support < n)
        if informative.sum() < 2 or float(intensity[informative].sum()) < .05:
            return None
        return (credits * information[None,:]) @ intensity

    def score(self, spectra, candidates, deadline):
        module = self.scope.get('frag_rescore')
        if module is None or time.monotonic() >= deadline:
            return {}
        try:
            for smile in candidates:
                graph = module.mol_graph(smile)
                if graph is None or len(graph[1]) > 40:
                    return {}
            cfg = module.resolve_cfg(dict(max_breaks=2))
            levels = [module.candidate_levels(s,cfg,self.cache) for s in candidates]
            if any(level is None for level in levels):
                return {}
            scores = []
            for spectrum in spectra:
                prep = module._prep_spectrum(spectrum,cfg)
                if prep is None:
                    continue
                iw,q,ns,npk = prep
                credits = np.zeros((len(candidates),npk))
                for i,level in enumerate(levels):
                    for masses,weight in zip(level,(1.,cfg['w1'],cfg['w2'])):
                        if len(masses):
                            hits = (np.searchsorted(masses,q+cfg['tol'],'right') >
                                np.searchsorted(masses,q-cfg['tol'],'left')).reshape(ns,npk).any(axis=0)
                            credits[i] = np.maximum(credits[i],hits*weight)
                result = self.contrast(credits,iw)
                if result is not None:
                    scores.append(result)
            if not scores:
                return {}
            values = np.mean(scores,axis=0)
            if not np.isfinite(values).all() or np.ptp(values) <= 1e-8:
                return {}
            return dict(zip(candidates,map(float,values)))
        except Exception as error:
            self.errors.append(repr(error))
            return {}

    def run(self):
        scope, control = self.scope, self.control()
        self.cache, self.errors, self.audit = {}, [], []
        deadline = time.monotonic() + (900 if scope['IS_RERUN'] else 120)
        frames = {str(mid):frame for mid,frame in scope['te'].groupby('molecule_id',sort=False)}
        rows, accepted = [], 0
        for row in control.itertuples(index=False):
            mid,original = row.molecule_id,row.smiles.split(';')
            revised = list(original)
            ice,gl = scope['ICE_SCORES'].get(mid,{}),scope['gl_of'](mid)
            if mid in frames and not gl and not any(v is not None for v in ice.values()):
                c,s,k,f = self.fused(mid)
                baseline = self.baseline_order(mid,c,s,k,f)
                current = [c[i] for i in baseline]
                assert scope['safe_extend'](current[:25],current[25:]+original) == original
                spectra = [dict(mz=np.asarray(r.ms2_mzs),it=np.asarray(r.ms2_normalized_intensities),
                    adduct=r.adduct,prec=float(r.precursor_mz),mode=1 if r.ionization_mode=='positive' else -1)
                    for r in frames[mid].itertuples(index=False)][:16]
                groups = {}
                for i,formula in enumerate(f):
                    groups.setdefault(formula,[]).append(i)
                evidence,eligible = {},set()
                positions = {i:j for j,i in enumerate(baseline)}
                for formula,indices in groups.items():
                    if not 2 <= len(indices) <= 8:
                        continue
                    complete = self.score(spectra,[c[i] for i in indices],deadline)
                    self.audit.append(dict(molecule_id=mid,formula=formula,candidates=len(indices),accepted=bool(complete)))
                    if complete:
                        evidence.update(complete)
                        eligible.update(positions[i] for i in indices)
                        accepted += 1
                if evidence:
                    by_smile = {smile:i for i,smile in enumerate(c)}
                    aligned = [by_smile[smile] for smile in current]
                    order = scope['gl_fuse'].rerank_multi(current,[k[i] for i in aligned],
                        [s[i] for i in aligned],[f[i] for i in aligned],[evidence],[1.25],top_n=len(current))
                    self.preserve_slots(order,eligible)
                    current = [current[i] for i in order]
                    revised = scope['safe_extend'](current[:25],current[25:]+original)
            rows.append((mid,';'.join(revised)))
        return self.finish(rows,control,'discriminative_fragment_fallback',dict(target_groups=len(self.audit),
            accepted_groups=accepted,errors=self.errors,max_group_size=8,max_bonds=40,contrast_lambda=1.25,
            budget_seconds=900 if scope['IS_RERUN'] else 120))
'''


def build():
    base = initial_build()[0]
    outputs = []
    for number,title,method,classes,launch in (
        (5,'Simulator Conflict Check',
         'Only complete, positive, uniquely winning GLACIER evidence is considered. Its leader must disagree '
         'with both the fusion leader and the completed baseline group leader. ICEBERG checks the entire '
         'same-formula group (2..8 candidates) using supported positive H/Na spectra. Promote only the '
         'same unique winner, swapping two slots inside that group. Missing or tied evidence rolls back. '
         'Reuse a complete ICE pass; otherwise run one new complete-group pass with 1800s hidden / 120s smoke budget.',
         COVERAGE+CONSENSUS,'SimulatorConflictCheck'),
        (6,'Discriminative Fragment Evidence',
         'Use the v7c no-forward-evidence molecule gate. Compare 2..8 same-formula candidates with at most '
         '40 bonds each. Keep upstream two-cleavage masses, adduct carriers, hydrogen shifts and tolerance. '
         'For each observed peak multiply its cleavage credit by log(N/support)/log(N), where support is '
         'the number of candidates that explain this peak. Shared peaks receive zero contrast weight; '
         'precursor peaks are not explicitly removed. Require at least two informative peaks carrying '
         '5% of spectrum intensity. Mean informative-spectrum evidence replaces (not stacks onto) the '
         'baseline fragment term at lambda 1.25. Unlike lambda 0.5, this can reverse a two-candidate '
         'group after within-group z-normalization; it is deliberately aggressive, not calibrated. '
         'Other groups retain their slots. Budget 900s hidden / 120s smoke.',
         DISCRIMINATIVE,'DiscriminativeFragmentFallback')):
        nb = copy.deepcopy(base)
        nb['cells'][0] = code_cell(prose('CASMI26: Closing the Evidence Gap\nIteration %02d — %s\n'%(number,title)
            + 'Independent hypothesis on the user-confirmed 0.441 Initial01 baseline. No measured score for this version.\n'
            + method + '\nSame inputs as the successful 0.441 run, including RDKit 2025.3.6 cp313 for Python 3.13. '
            + 'Keep the separate main-engine RDKit 2026 wheel input. Importing code does not guarantee Kaggle attaches inputs.'))
        nb['cells'].append(code_cell(prose('Method and attribution\n'+method+'\n'
            + 'Our contribution: conditional simulator agreement / candidate-discriminative peak weighting. '
            + 'This is not a claim of independent predictor or fragment-kernel authorship. '
            + 'The full upstream credits and links are retained above. Correlated simulator errors and '
            + 'fragment over-explanation can reduce MRR. Same-run parity assertions preserve the control. '
            + 'Read iteration_audit.json and iteration_vs_0441.csv before submitting; a no-op is not a gain. '
            + 'Initial01 plots describe the control, not this treatment.')
            + COMMON+classes+f'iteration = {launch}(experiment)\niteration_result = iteration.run()\n'))
        nb['metadata']['casmi_experiment'].update(iteration=launch,measured_score=None,
            user_reported_baseline_score=.441,validation='Synthetic boundary tests; full Kaggle execution pending')
        validate(nb)
        outputs.append(nb)
    return outputs


if __name__ == '__main__':
    for name,nb in zip(('casmi26_iteration05_simulator_conflict.ipynb','casmi26_iteration06_discriminative_fragments.ipynb'),build()):
        path = ROOT/'research/evidence-routing'/name
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
