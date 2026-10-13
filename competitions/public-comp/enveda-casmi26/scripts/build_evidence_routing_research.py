'''Initial research notebooks: source-verified OOP routing and isolated group-gap treatment.'''
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path
from build_flow_experiments import ROOT, prose, validate
from build_metric_safe_adaptation import code_cell

UPLOAD = Path('/Users/flexonafft/Downloads/casmi26-v7c-e006-fraggap.ipynb')
SOURCE = ROOT / 'archive/public-sources-2026-10-13/casmi26_v7c_uploaded.ipynb'

FRAMEWORK = '''
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, math, time
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ExperimentConfig:
    group_fragment_fallback: bool = False
    window: int = 40
    fragment_weight: float = 0.5

class SourceStage:
    def __init__(self, label, source_cells):
        self.label, self.source_cells = label, source_cells

    def run(self, scope):
        started = time.time()
        print('STAGE:', self.label, flush=True)
        for index, source in self.source_cells:
            exec(compile(source, f'<credited-v7c-cell-{index}>', 'exec'), scope)
        return dict(stage=self.label, seconds=round(time.time() - started, 2))

class InputRegistry(SourceStage):
    def run(self, scope):
        import sys
        if sys.version_info[:2] not in ((3, 12), (3, 13)):
            raise RuntimeError(
                f'Supported runtimes: Python 3.12/3.13; got {sys.version_info.major}.{sys.version_info.minor}. '
                'Use a compatible runtime and RDKit 2025.3.6 simulator wheel; never substitute RDKit 2026.')
        return super().run(scope)

class CandidateRetriever(SourceStage):
    pass

class ForwardEvidence(SourceStage):
    pass

class MetricAudit(SourceStage):
    pass

class FragmentEvidenceScorer(SourceStage):
    def __init__(self, label, source_cells):
        super().__init__(label, source_cells)
        self.cache, self.errors = {}, []

    def score(self, scope, mid, candidates):
        module = scope.get('frag_rescore')
        if module is None or '_fcache' not in scope:
            return {}
        frame = scope['te'][scope['te'].molecule_id.astype(str) == str(mid)]
        spectra = [dict(mz=np.asarray(r.ms2_mzs, np.float64),
            it=np.asarray(r.ms2_normalized_intensities, np.float64), adduct=r.adduct,
            prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1)
            for r in frame.itertuples(index=False)][:16]
        if not spectra:
            return {}
        try:
            values = module.score_molecule(spectra, candidates, cache=self.cache)
            return {s: float(v) for s, v in zip(candidates, values) if v is not None and math.isfinite(float(v))}
        except Exception as error:
            self.errors.append(dict(molecule_id=str(mid), error=repr(error)))
            print('GROUP FRAGMENT FAILED: preserve v7c order', mid, repr(error))
            return {}

class MetricIdentityFusion:
    def __init__(self, scope):
        self.scope = scope

    def rank(self, first, second, window):
        return self.scope['metric_fusion'](first, second, self.scope['metric_identity'],
            alpha=self.scope['ALPHA'], offset=self.scope['KRR'], limit=window)

class EvidenceRouter:
    def __init__(self, scope, config, fragments):
        self.scope, self.config, self.fragments = scope, config, fragments
        self.fusion = MetricIdentityFusion(scope)

    @staticmethod
    def coverage(scores, candidates, keys):
        count = 0
        for smile, key in zip(candidates, keys):
            value = scores.get(smile)
            if value is None:
                value = scores.get(key)
            try:
                count += value is not None and math.isfinite(float(value))
            except (TypeError, ValueError):
                pass
        return count

    def group_rescore(self, mid, candidates, original_candidates, scores, ice, gl):
        scope = self.scope
        keys = [scope['chem'].score_key(s) or s for s in candidates]
        formulas = [scope['_form2'](s) for s in candidates]
        groups = {}
        for index, formula in enumerate(formulas):
            groups.setdefault(formula, []).append(index)
        missing = []
        for indices in groups.values():
            smiles = [candidates[i] for i in indices]
            group_keys = [keys[i] for i in indices]
            if len(indices) >= 2 and not self.coverage(ice, smiles, group_keys) and not self.coverage(gl, smiles, group_keys):
                missing.append(indices)
        if not missing:
            return list(candidates), 0
        needed = [candidates[i] for indices in missing for i in indices]
        evidence = self.fragments.score(scope, mid, needed)
        accepted = {}
        for indices in missing:
            if all(candidates[i] in evidence for i in indices):
                accepted.update((candidates[i], evidence[candidates[i]]) for i in indices)
        if not accepted:
            return list(candidates), 0
        score_by_smile = dict(zip(original_candidates, scores))
        aligned_scores = [score_by_smile[s] for s in candidates]
        order = scope['gl_fuse'].rerank_multi(candidates, keys, aligned_scores, formulas,
            [accepted], [self.config.fragment_weight], top_n=len(candidates))
        # Group-slot preservation is a trust boundary, not just a plotting assumption.
        eligible_positions = {i for indices in missing if all(candidates[j] in accepted for j in indices) for i in indices}
        assert all(i == j for i, j in enumerate(order) if i not in eligible_positions)
        return [candidates[i] for i in order], sum(all(candidates[i] in accepted for i in indices) for indices in missing)
'''

REPORTING = '''
class SubmissionBuilder:
    def __init__(self, scope):
        self.scope = scope

    def validate(self, frame, control, active):
        assert list(frame.columns) == ['molecule_id', 'smiles']
        assert list(frame.molecule_id) == list(control.molecule_id) and frame.molecule_id.is_unique
        assert frame.smiles.notna().all()
        original = dict(zip(control.molecule_id, control.smiles))
        for row in frame.itertuples(index=False):
            smiles = row.smiles.split(';')
            assert 1 <= len(smiles) <= 25
            if row.molecule_id in active:
                identities = list(map(self.scope['metric_identity'], smiles))
                assert None not in identities and len(identities) == len(set(identities))
            else:
                assert row.smiles == original[row.molecule_id]

class ExperimentAudit:
    def save(self, audit, stages, fragments, config):
        frame = pd.DataFrame(audit)
        frame.to_csv('research_routing_audit.csv', index=False)
        active = frame[frame.processed]
        summary = dict(experiment='group-gap' if config.group_fragment_fallback else 'initial-faithful',
            active_molecules=len(active), output_rows=len(frame),
            source_control_reconstructed=True, changed_rows=int(frame.changed.sum()),
            changed_top1=int(frame.changed_top1.sum()), fragment_errors=fragments.errors,
            source_control_sha256=hashlib.sha256(Path('submission_v7c_control.csv').read_bytes()).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
            group_fallback_count=int(frame.group_fragment_groups.sum()), stages=stages,
            routes=active.branch.value_counts().to_dict(), measured_score=None,
            validation_note='Visible smoke is not independent validation; source public 0.441 is user-reported.')
        Path('research_routing_audit.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        if config.group_fragment_fallback and not summary['changed_rows']:
            print('NO-OP versus v7c: do not spend another submission on identical predictions.')
        self.plot(active)
        return summary

    def plot(self, active):
        try:
            import matplotlib.pyplot as plt
            counts = active.branch.value_counts().sort_values()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].barh(counts.index, counts.values, color='#35618c')
            for i, value in enumerate(counts.values):
                axes[0].text(value, i, f' {value}', va='center')
            axes[0].set(title='Final routing branch — active molecules', xlabel='Molecules')
            changes = [int(active.changed.sum()), int(active.changed_top1.sum())]
            axes[1].bar(['Changed lists', 'Changed top-1'], changes, color='#a17e35')
            for i, value in enumerate(changes):
                axes[1].text(i, value, str(value), ha='center', va='bottom')
            axes[1].set(title='Changes versus same-run v7c', ylabel='Molecules')
            for axis in axes:
                axis.set_axisbelow(True)
                axis.grid(axis='x' if axis is axes[0] else 'y', alpha=.15)
                axis.spines[['top', 'right']].set_visible(False)
            axes[0].set_xlim(0, max(1, int(counts.max()) if len(counts) else 1) * 1.18)
            axes[1].set_ylim(0, max(1, max(changes)) * 1.2)
            fig.suptitle('Current-run audit, not a validation score', fontsize=12)
            fig.tight_layout()
            fig.savefig('research_evidence_report.png', dpi=150, bbox_inches='tight')
            plt.show()
            plt.close(fig)
        except ImportError:
            print('Chart dependency unavailable; inspect research_routing_audit.csv/json.')

class CasmiExperiment:
    def __init__(self, config, stages, fragments, reference_route):
        self.config, self.stages, self.fragments = config, stages, fragments
        self.reference_route = reference_route
        try:
            from IPython.display import display
        except ImportError:
            display = print
        self.scope = {'__name__': '__main__', 'display': display}

    def run(self):
        scope = self.scope
        timing = [stage.run(scope) for stage in self.stages]
        timing.append(self.fragments.run(scope))
        timing.append(self.reference_route.run(scope))
        Path('submission_v7c_control.csv').write_bytes(Path('submission.csv').read_bytes())
        control = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
        reference = pd.read_csv('submission_reference.csv', dtype=str, keep_default_na=False)
        assert list(control.molecule_id) == list(reference.molecule_id)
        controls = dict(zip(control.molecule_id, control.smiles))
        active = set(scope['te'].molecule_id.astype(str))
        router = EvidenceRouter(scope, self.config, self.fragments)
        rows, audits = [], []
        for row in reference.itertuples(index=False):
            mid = row.molecule_id
            original = row.smiles.split(';')
            faithful, _ = router.rerank_row(mid, original, window=40, ice_fallback=True, processed=mid in active, group=False)
            assert ';'.join(faithful) == controls[mid], f'OOP source parity mismatch: {mid}'
            revised, audit = router.rerank_row(mid, original, window=self.config.window,
                ice_fallback=True, processed=mid in active, group=self.config.group_fragment_fallback)
            old = controls[mid].split(';')
            audit.update(changed=revised != old, changed_top1=revised[:1] != old[:1])
            rows.append((mid, ';'.join(revised)))
            audits.append(audit)
        output = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
        SubmissionBuilder(scope).validate(output, control, active)
        output.to_csv('submission.csv', index=False)
        if not self.config.group_fragment_fallback:
            assert output.equals(control)
        return ExperimentAudit().save(audits, timing, self.fragments, self.config)
'''


def router_method(original):
    tree = ast.parse(original)
    method = next(n for n in tree.body[0].body if isinstance(n, ast.FunctionDef) and n.name == 'rerank_row')
    method = copy.deepcopy(method)
    method.args.args[0].arg = 'self'
    method.args.kwonlyargs.append(ast.arg(arg='group'))
    method.args.kw_defaults.append(ast.Constant(False))
    method.body.insert(0, ast.parse('scope = self.scope').body[0])
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == '_finite_coverage':
                node.func = ast.Attribute(ast.Name('self', ast.Load()), 'coverage', ast.Load())
            return node
    method = Rewrite().visit(method)
    value = ast.unparse(ast.fix_missing_locations(method))
    value = value.replace("'fallback_changed_top1': False}", "'fallback_changed_top1': False, 'group_fragment_groups': 0}")
    assert "'group_fragment_groups'" in value
    anchor = '    result = scope[\'safe_extend\']'
    assert anchor in value
    value = value.replace(anchor,
        "    if group and audit['branch'] != 'frag_gap':\n"
        "        candidates, ng = self.group_rescore(mid, candidates, before, scores, ice, gl)\n"
        "        audit['group_fragment_groups'] = ng\n"
        "        if ng:\n"
        "            audit['branch'] += '+group_fragment'\n" + anchor, 1)
    # The numerical source fusion function is reused through a named object boundary.
    value = value.replace("scope['metric_fusion'](first, engine['smiles'], scope['metric_identity'], alpha=scope['ALPHA'], offset=scope['KRR'], limit=window)",
                          "self.fusion.rank(first, engine['smiles'], window)")
    return '\n'.join('    ' + line if line else '' for line in value.splitlines()) + '\n'


def build():
    original = json.loads(SOURCE.read_text())
    sources = {i: ''.join(c['source']) for i, c in enumerate(original['cells']) if c['cell_type'] == 'code'}
    old_guard = "assert platform.python_version_tuple()[:2] == ('3', '12')"
    assert sources[8].count(old_guard) == 1
    sources[8] = sources[8].replace(old_guard,
        "assert platform.python_version_tuple()[:2] in (('3', '12'), ('3', '13')), 'Python 3.12/3.13 required'", 1)
    base_credits = sources[3]
    setup = '''
stages = [
    InputRegistry('Runtime and verified inputs', [(i, SOURCE_CELLS[i]) for i in (7, 8, 10)]),
    CandidateRetriever('Retrieval, fingerprints and base rankings', [(i, SOURCE_CELLS[i]) for i in (12, 14, 16, 17, 19)]),
    ForwardEvidence('Simulators, PubChem gate and inherited fusion', [(i, SOURCE_CELLS[i]) for i in (21, 23, 25)]),
    MetricAudit('Official identity audit and same-run source observer', [(i, SOURCE_CELLS[i]) for i in (27, 28, 29, 30)]),
]
fragments = FragmentEvidenceScorer('MetFrag-lite molecule-gap reference', [(32, SOURCE_CELLS[32])])
reference_route = SourceStage('Unmodified v7c routing oracle', [(33, SOURCE_CELLS[33])])
'''
    notebooks = []
    for experiment in (False, True):
        label = 'Initial 02 — Group-Level Fragment Fallback' if experiment else 'Initial 01 — Source-Verified OOP Adaptation'
        cells = [code_cell(prose('CASMI26: Evidence Routing & Fragment Fallback\n' + label + '\n'
            'Research question: what evidence should rank candidates when forward simulators leave a gap? '
            'Starting source: uploaded v7c, public 0.441 reported by the user; E006 0.440 is our confirmed '
            'baseline. Neither initial version has a measured score yet. '
            'Authors: @ahmedberatozer (v7c header attribution; uploaded file has no verified direct notebook URL), '
            '@canqiang (E006), @flexonafft (metric fusion), @samanyu1808 (ICE fallback), '
            'Huseyin Emre Aksoy (fragment-rescorer), and the upstream contributors credited below.\n'
            'Python 3.12 or 3.13, GPU T4, Internet OFF. Retain the uploaded source inputs. '
            'For Python 3.13 also attach https://www.kaggle.com/datasets/eveliaveldrine/rdkit-wheels '
            '(RDKit 2025.3.6 cp313 Linux x86_64). Keep metric/rdkit-2026-3-3-wheel for the main '
            'RDKit 2026.03.3 environment. The existing runtime-tagged selector installs RDKit 2025.3.6 '
            'only into the isolated simulator site. Both Python guards now accept 3.12/3.13; '
            'all wheel, GPU and RDKit checks remain enabled. Python 3.13 score parity is unverified. '
            'Source numerical kernels and subprocess modules are preserved. Our author-written orchestration, '
            'routing, group fallback, submission checks and reporting are object-oriented. '
            'Credited source stages remain procedural adapters, not a claim that upstream code was rewritten.')),
            code_cell(prose('Credits, licences and contribution boundaries\n'
                'Direct source snapshot: ' + SOURCE.name + '; SHA256: ' + hashlib.sha256(SOURCE.read_bytes()).hexdigest() + '\n'
                '@ahmedberatozer — v4n engine/resources and v7c source attribution: https://www.kaggle.com/ahmedberatozer\n'
                '@canqiang — E006 reproduction: https://www.kaggle.com/canqiang\n'
                '@samanyu1808 — fallback idea: https://www.kaggle.com/code/samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438\n'
                'Huseyin Emre Aksoy — reused frag_rescore.py, described as Apache-2.0 in the uploaded source: '
                'https://www.kaggle.com/huseyinemreaksoy\n'
                '@lszlst — GLACIER-only ablations; @skomuro — promotion rule; @megayak — Engine2 assets, '
                'as credited by the v7c source. https://www.kaggle.com/lszlst ; https://www.kaggle.com/skomuro ; '
                'https://www.kaggle.com/megayak\n'
                'This work reuses substantial code verbatim. OOP organization does not transfer authorship '
                'or override component licences. New contribution: explicit routing objects, source-parity '
                'assertions, paired audits, and the isolated group-gap hypothesis. No labels or network calls.') + base_credits),
            code_cell(prose('1. Method and controls\n'
                'Pipeline: verified inputs -> two-engine retrieval -> metric-identity fusion -> '
                'GLACIER / ICEBERG / fragment evidence -> audited top-25 submission. '
                'Initial 01 preserves the exact v7c policy and verifies OOP output against its same-run '
                'source output. Initial 02 starts from that identical route, then fills fully uncovered '
                'same-formula groups even inside partly simulator-covered molecules. '
                'Any finite simulator score in a group excludes that group; groups need at least two '
                'candidates and complete fragment scores. Already molecule-fragment-reranked lists '
                'are not reranked twice. Covered groups keep their slots. Weight 0.5, window40, all '
                'upstream parameters unchanged. The group variant can change top-1 and lower MRR.')),
            code_cell(prose('2. Configuration, credited adapters and routing objects\n'
                'The adapters execute preserved source cells in one explicit per-experiment namespace '
                'to keep upstream imports and subprocess contracts intact. Kernels are not hidden claims '
                'of original authorship. Classes own the new routing and audit boundaries.') + FRAMEWORK + router_method(sources[33])),
            code_cell(prose('3. Submission integrity and current-run evidence report\n'
                'The charts describe actual processed molecules, not model accuracy. Placeholder rows '
                'are excluded. The paired reference is submission_v7c_control.csv; raw per-molecule '
                'routes and changes remain inspectable in research_routing_audit.csv/json.') + REPORTING),
            code_cell(prose('4. Preserved numerical source pack\n'
                'Source arithmetic and embedded upstream modules are kept verbatim. The source preflight '
                'Python-version assertion alone is widened to 3.12/3.13 for compatible simulator wheels. '
                'Inspect SOURCE_CELLS[index] for the exact implementation; the source snapshot hash '
                'above identifies its provenance. Source cells are executed only by the ordered stages.')
                + 'SOURCE_CELLS = ' + repr({i: sources[i] for i in (7,8,10,12,14,16,17,19,21,23,25,27,28,29,30,32,33)}) + '\n'),
            code_cell(prose('5. Run the controlled experiment\n'
                'This runs the existing neural/model recipe once, then reconstructs routing without '
                'another neural pass. Source-parity mismatches stop execution before writing the new '
                'submission. Full inference and chart outputs require the Kaggle inputs/GPU environment.')
                + setup + f'config = ExperimentConfig(group_fragment_fallback={experiment!r})\n'
                + 'experiment = CasmiExperiment(config, stages, fragments, reference_route)\n'
                + 'result = experiment.run()\n'),
            code_cell(prose('6. Interpretation and next decision\n'
                'Use the official scored rerun to evaluate this experiment. A changed list is not an '
                'improved list. A smoke-run no-op does not establish hidden-set inactivity. '
                'The source public 0.441 is not our reproduced result. Keep source-verified Initial 01 '
                'separate from Initial 02; do not stack preceding unsuccessful experiments. '
                'Publish exact run settings, measured results, author links and caveats rather than '
                'treating OOP or source appearance as originality.'))]
        nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
            metadata={'kernelspec':original['metadata']['kernelspec'], 'kaggle':dict(original['metadata']['kaggle'], isGpuEnabled=True),
                'casmi_experiment':dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    user_reported_source_score=.441, measured_score=None, group_fragment_fallback=experiment,
                    compatibility_fix='Both Python guards allow 3.12/3.13; runtime-tagged simulator RDKit selector preserved',
                    validation='Synthetic source-parity and group-slot tests; full Kaggle run pending')})
        validate(nb)
        notebooks.append(nb)
    return notebooks


if __name__ == '__main__':
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        shutil.copy2(UPLOAD, SOURCE)
    destination = ROOT / 'research/evidence-routing'
    destination.mkdir(parents=True, exist_ok=True)
    for name, nb in zip(('casmi26_evidence_routing_initial01.ipynb', 'casmi26_evidence_routing_initial02_group_gap.ipynb'), build()):
        path = destination / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(path)
