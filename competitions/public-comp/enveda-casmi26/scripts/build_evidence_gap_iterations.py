'''Two isolated OOP evidence experiments on confirmed Initial 01 / v7c 0.441.'''
import copy
import json
from build_evidence_routing_research import build as initial_build
from build_flow_experiments import ROOT, prose, validate
from build_metric_safe_adaptation import code_cell

COMMON = '''
class EvidenceIteration:
    def __init__(self, experiment):
        self.experiment, self.scope = experiment, experiment.scope
        self.audit = []

    @staticmethod
    def finite(value):
        try:
            return value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def value(self, evidence, smile, key):
        value = evidence.get(smile)
        return evidence.get(key) if value is None else value

    def fused(self, mid):
        scope = self.scope
        engine = scope['ENG'].get(mid)
        if not engine or not engine.get('smiles') or mid not in scope['PREFUSION_LISTS']:
            return [], [], [], []
        first = [s for s in scope['PREFUSION_LISTS'][mid] if s and s != 'CCO']
        candidates, scores, _ = MetricIdentityFusion(scope).rank(first, engine['smiles'], 40)
        keys = [scope['chem'].score_key(s) or s for s in candidates]
        formulas = [scope['_form2'](s) for s in candidates]
        return candidates, scores, keys, formulas

    def baseline_order(self, mid, candidates, scores, keys, formulas):
        scope = self.scope
        ice, gl = scope['ICE_SCORES'].get(mid, {}), scope['gl_of'](mid)
        if ice and gl:
            return scope['gl_fuse'].rerank_multi(candidates, keys, scores, formulas,
                [{}, gl], [scope['ICE_LAM'], scope['GL_LAM']], top_n=len(candidates))
        frag = scope.get('FRAG_GAP', {}).get(mid)
        if not gl and frag and not any(v is not None for v in (ice or {}).values()):
            return scope['gl_fuse'].rerank_multi(candidates, keys, scores, formulas,
                [frag], [.5], top_n=len(candidates))
        if ice and not gl:
            return scope['ice_fuse'].rerank(candidates, keys, scores, formulas, ice,
                lam=scope['ICE_LAM'], top_n=len(candidates))
        return list(range(len(candidates)))

    def preserve_slots(self, order, eligible):
        assert sorted(order) == list(range(len(order)))
        assert all(i == j for i,j in enumerate(order) if i not in eligible)

    def finish(self, rows, control, name, details):
        frame = pd.DataFrame(rows, columns=['molecule_id','smiles'])
        active = set(self.scope['te'].molecule_id.astype(str))
        SubmissionBuilder(self.scope).validate(frame, control, active)
        frame.to_csv('submission.csv', index=False)
        comparison = pd.DataFrame(dict(molecule_id=control.molecule_id,
            changed=control.smiles != frame.smiles,
            changed_top1=control.smiles.str.split(';').str[0] != frame.smiles.str.split(';').str[0]))
        comparison.to_csv('iteration_vs_0441.csv', index=False)
        pd.DataFrame(self.audit).to_csv('iteration_group_audit.csv', index=False)
        report = dict(experiment=name, changed_rows=int(comparison.changed.sum()),
            changed_top1=int(comparison.changed_top1.sum()), details=details, measured_score=None,
            control_sha256=hashlib.sha256(Path('submission_0441_control.csv').read_bytes()).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest())
        Path('iteration_audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not report['changed_rows']:
            print('NO-OP versus Initial 01: do not spend a second submission on identical predictions.')
        return report

    def control(self):
        Path('submission_0441_control.csv').write_bytes(Path('submission.csv').read_bytes())
        return pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
'''

COVERAGE = '''
class FusionFirstCoverage(EvidenceIteration):
    def prepare(self, control):
        scope = self.scope
        self.audit = []
        self.plans, self.targets = {}, {'gl':{}, 'ice':{}}
        frames = {str(mid):frame for mid,frame in scope['te'].groupby('molecule_id',sort=False)}
        for row in control.itertuples(index=False):
            mid = row.molecule_id
            if mid not in frames:
                continue
            candidates, scores, keys, formulas = self.fused(mid)
            if not candidates:
                continue
            order = self.baseline_order(mid, candidates, scores, keys, formulas)
            ordered = [candidates[i] for i in order]
            assert scope['safe_extend'](ordered[:25],ordered[25:]+row.smiles.split(';')) == row.smiles.split(';'), 'Initial 01 reconstruction differs'
            self.plans[mid] = (candidates,scores,keys,formulas,order)
            ice, gl = scope['ICE_SCORES'].get(mid,{}), scope['gl_of'](mid)
            supported_gl = bool(((frames[mid].ionization_mode == 'positive') & (frames[mid].adduct == '[M+H]+')).any())
            supported_ice = bool(((frames[mid].ionization_mode == 'positive') & frames[mid].adduct.isin(['[M+H]+','[M+Na]+'])).any())
            stage = 'gl' if gl or (supported_gl and not ice) else 'ice'
            supported = supported_gl if stage == 'gl' else supported_ice
            evidence = gl if stage == 'gl' else ice
            groups = {}
            for i, formula in enumerate(formulas):
                groups.setdefault(formula,[]).append(i)
            for formula,indices in groups.items():
                if len(indices) < 2:
                    continue
                missing = sum(not self.finite(self.value(evidence,candidates[i],keys[i])) for i in indices)
                if not missing:
                    continue
                self.audit.append(dict(molecule_id=mid, formula=formula, stage=stage,
                    candidates=len(indices), missing_scores=missing, supported=supported, accepted=False))
                if supported:
                    self.targets[stage].setdefault(mid,[]).append(indices)
        return frames

    def predict(self):
        scope = self.scope
        output = {'gl':{},'ice':{}}
        budget = 1800 if scope['IS_RERUN'] else 120
        for stage in ('gl','ice'):
            candidate_map = {}
            for mid,groups in self.targets[stage].items():
                candidates = self.plans[mid][0]
                candidate_map[mid] = [candidates[i] for indices in groups for i in indices]
            if not candidate_map:
                continue
            items = scope['ice_fuse'].build_ice_input(scope['te'], candidate_map)
            if stage == 'gl':
                items = [dict(item,spectra=[s for s in item['spectra'] if s.get('adduct')=='[M+H]+']) for item in items]
            items = [item for item in items if item['spectra']]
            workdir = '/kaggle/working/fusion_coverage_' + stage
            if stage == 'gl':
                output[stage] = scope['gl_fuse'].run_gl(scope['GL_PKG'], items, workdir=workdir,
                    device='cuda', budget_s=budget, site='/kaggle/working/ice_site',
                    wheels=str(Path(scope['ICE_PKG'])/'wheels'), extra_args=['--any-rdkit'])
            else:
                output[stage] = scope['ice_fuse'].run_ice(scope['ICE_PKG'], items, workdir=workdir,
                    device='cuda', budget_s=budget, site='/kaggle/working/ice_site', extra_args=['--any-rdkit'])
        return output

    def complete_group(self, indices, candidates, keys, original, predicted):
        combined = {}
        for i in indices:
            value = self.value(original,candidates[i],keys[i])
            if not self.finite(value):
                value = self.value(predicted,candidates[i],keys[i])
            if not self.finite(value):
                return {}
            combined[candidates[i]] = float(value)
        return combined

    def run(self):
        control = self.control()
        self.prepare(control)
        predicted = self.predict()
        rows, accepted_groups = [], 0
        for row in control.itertuples(index=False):
            mid, original = row.molecule_id, row.smiles.split(';')
            revised = list(original)
            if mid in self.plans:
                candidates,scores,keys,formulas,baseline_order = self.plans[mid]
                current = [candidates[i] for i in baseline_order]
                positions = {i:j for j,i in enumerate(baseline_order)}
                for stage in ('gl','ice'):
                    old = self.scope['gl_of'](mid) if stage=='gl' else self.scope['ICE_SCORES'].get(mid,{})
                    evidence, eligible = {}, set()
                    for indices in self.targets[stage].get(mid,[]):
                        complete = self.complete_group(indices,candidates,keys,old,predicted[stage].get(mid,{}))
                        if complete:
                            evidence.update(complete)
                            eligible.update(positions[i] for i in indices)
                            accepted_groups += 1
                            formula = formulas[indices[0]]
                            next(a for a in self.audit if a['molecule_id']==mid and a['formula']==formula)['accepted'] = True
                    if evidence:
                        by_smile = dict(zip(candidates,scores))
                        key_by_smile = dict(zip(candidates,keys))
                        form_by_smile = dict(zip(candidates,formulas))
                        order = self.scope['gl_fuse'].rerank_multi(current,[key_by_smile[s] for s in current],
                            [by_smile[s] for s in current],[form_by_smile[s] for s in current],
                            [evidence],[self.scope['GL_LAM'] if stage=='gl' else self.scope['ICE_LAM']],top_n=len(current))
                        self.preserve_slots(order,eligible)
                        current = [current[i] for i in order]
                if any(a['accepted'] for a in self.audit if a['molecule_id']==mid):
                    revised = self.scope['safe_extend'](current[:25],current[25:]+original)
            rows.append((mid,';'.join(revised)))
        return self.finish(rows,control,'fusion_first_coverage',dict(target_groups=len(self.audit),
            accepted_groups=accepted_groups, supported_groups=sum(a['supported'] for a in self.audit),
            per_simulator_budget_seconds=1800 if self.scope['IS_RERUN'] else 120))
'''

THREE = '''
class ThreeBondFragmentFallback(EvidenceIteration):
    def score(self, spectra, candidates, budget_deadline):
        module = self.scope.get('frag_rescore')
        if module is None or '_fcache' not in self.scope or time.monotonic() >= budget_deadline:
            return {}
        try:
            for smile in candidates:
                graph = module.mol_graph(smile)
                if graph is None or len(graph[1]) > 24:
                    return {}
            values = module.score_molecule(spectra,candidates,
                cfg=dict(max_breaks=3,max_bonds_3=24,w2=.6,w3=.3),cache=self.cache)
            if len(values) != len(candidates) or not all(self.finite(v) for v in values):
                return {}
            return dict(zip(candidates,map(float,values)))
        except Exception as error:
            self.errors.append(repr(error))
            return {}

    def run(self):
        scope = self.scope
        control = self.control()
        self.audit = []
        self.cache, self.errors = {}, []
        deadline = time.monotonic() + (900 if scope['IS_RERUN'] else 120)
        frames = {str(mid):frame for mid,frame in scope['te'].groupby('molecule_id',sort=False)}
        rows, accepted_groups = [], 0
        for row in control.itertuples(index=False):
            mid, original = row.molecule_id,row.smiles.split(';')
            revised = list(original)
            ice,gl = scope['ICE_SCORES'].get(mid,{}),scope['gl_of'](mid)
            # Keep the v7c molecule-level eligibility, not the unproven group-gap treatment.
            if mid in frames and not gl and not any(v is not None for v in (ice or {}).values()):
                candidates,scores,keys,formulas = self.fused(mid)
                baseline_order = self.baseline_order(mid,candidates,scores,keys,formulas)
                current = [candidates[i] for i in baseline_order]
                assert scope['safe_extend'](current[:25],current[25:]+original) == original
                spectra = [dict(mz=np.asarray(r.ms2_mzs,np.float64),it=np.asarray(r.ms2_normalized_intensities,np.float64),
                    adduct=r.adduct,prec=float(r.precursor_mz),mode=1 if r.ionization_mode=='positive' else -1)
                    for r in frames[mid].itertuples(index=False)][:16]
                groups = {}
                for i,formula in enumerate(formulas):
                    groups.setdefault(formula,[]).append(i)
                evidence,eligible = {},set()
                positions = {i:j for j,i in enumerate(baseline_order)}
                for formula,indices in groups.items():
                    if len(indices)<2:
                        continue
                    group = [candidates[i] for i in indices]
                    complete = self.score(spectra,group,deadline)
                    self.audit.append(dict(molecule_id=mid,formula=formula,candidates=len(indices),accepted=bool(complete)))
                    if complete:
                        evidence.update(complete)
                        eligible.update(positions[i] for i in indices)
                        accepted_groups += 1
                if evidence:
                    scores_by_smile = dict(zip(candidates,scores))
                    keys_by_smile = dict(zip(candidates,keys))
                    forms_by_smile = dict(zip(candidates,formulas))
                    order = scope['gl_fuse'].rerank_multi(current,[keys_by_smile[s] for s in current],
                        [scores_by_smile[s] for s in current],[forms_by_smile[s] for s in current],
                        [evidence],[.5],top_n=len(current))
                    self.preserve_slots(order,eligible)
                    current = [current[i] for i in order]
                    revised = scope['safe_extend'](current[:25],current[25:]+original)
            rows.append((mid,';'.join(revised)))
        return self.finish(rows,control,'three_bond_fragment_fallback',dict(target_groups=len(self.audit),
            accepted_groups=accepted_groups, errors=self.errors, max_bonds_per_candidate=24,
            budget_seconds=900 if scope['IS_RERUN'] else 120, budget_checked_between_bounded_groups=True))
'''


def build():
    base = initial_build()[0]
    outputs = []
    for title, rationale, classes, launch in (
        ('Iteration 03 — Fusion-First Simulator Coverage',
         'Audit the actual metric-fusion top40. For competing same-formula groups with missing simulator '
         'scores, repeat the original predictor/scoring recipe on the complete comparison group. '
         'Keep existing finite scores; fill only missing values. Accept a group only with complete '
         'finite coverage, otherwise preserve its Initial01 order. Preserve every untargeted group slot. '
         'Prefer the existing GLACIER route, otherwise ICEBERG; fully unsupported spectra remain '
         'unchanged. A previously unavailable GLACIER group may be attempted when supported H spectra '
         'exist and ICE has no dictionary. Two extra inference budgets of at most 1800 seconds each '
         'plus runner grace/startup overhead. This is new evidence, not window expansion or new similarity.',
         COVERAGE, 'FusionFirstCoverage'),
        ('Iteration 04 — Three-Bond Fragment Fallback',
         'Keep the original molecule-level v7c eligibility: no GLACIER and no non-null ICE evidence. '
         'Extend MetFrag-lite enumeration to three cleavages for same-formula groups where EVERY '
         'candidate has at most 24 bonds. Two-cleavage credit remains 0.6, three-cleavage credit is '
         '0.3, reranking weight is 0.5. Keep original fusion scores; replace the two-cleavage '
         'fragment term instead of stacking it. Incomplete scores, large molecules or exhausted '
         'budget preserve the original group order. Budget 900 seconds, checked between bounded '
         'groups. More cleavage paths can explain noise and reduce MRR.', THREE, 'ThreeBondFragmentFallback')):
        nb = copy.deepcopy(base)
        nb['cells'][0] = code_cell(prose('CASMI26: Closing the Evidence Gap\n' + title + '\n'
            'Independent iteration on Initial01, user-confirmed public MRR 0.441. ' + rationale + '\n'
            'Run Initial01 first, including source/OOP parity assertions. Its byte-exact output is '
            'saved as submission_0441_control.csv. No preceding unsuccessful experiment is stacked. '
            'Same inputs and runtime-compatible RDKit wheels; no new datasets. No score is claimed.'))
        nb['cells'].append(code_cell(prose('Independent iteration: method, provenance and guardrails\n'
            + rationale + '\nOur contribution is the controlled coverage/deeper-fragment orchestration. '
            'The credited @ahmedberatozer predictor packages and Huseyin Emre Aksoy fragment kernels '
            'are reused without rewriting their numerical definitions. All authors, links and licences '
            'remain above. The iteration audit compares against completed Initial01 from the same run; '
            'the earlier research audit/graphs still describe Initial01. A smoke no-op is not hidden '
            'validation, but exact identical final predictions should not consume another submission.')
            + COMMON + classes + f'iteration = {launch}(experiment)\niteration_result = iteration.run()\n'))
        nb['metadata']['casmi_experiment'].update(iteration=launch,user_reported_baseline_score=.441,
            measured_score=None,validation='Synthetic paired routing and rollback tests; Kaggle inference pending')
        validate(nb)
        outputs.append(nb)
    return outputs


if __name__=='__main__':
    destination = ROOT/'research/evidence-routing'
    for name,nb in zip(('casmi26_iteration03_fusion_first_coverage.ipynb','casmi26_iteration04_three_bond_fragment.ipynb'),build()):
        path = destination/name
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
