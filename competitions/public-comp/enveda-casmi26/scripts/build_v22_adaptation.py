'''Source-preserving V22 adaptation and isolated metric-distinct runner-up hypothesis.'''
import copy
import hashlib
import json
from pathlib import Path
from build_flow_experiments import ROOT,prose,validate
from build_metric_safe_adaptation import code_cell

UPLOAD = Path('/Users/flexonafft/Downloads/casmi26-v22-rel150-e2r2-clean.ipynb')
SOURCE = ROOT/'archive/public-sources-2026-10-16/casmi26_v22_uploaded.ipynb'

FRAMEWORK = '''
from pathlib import Path
import glob
import hashlib
import json
import time
import pandas as pd

class V22Stage:
    def __init__(self,label,indices):
        self.label,self.indices = label,indices

    def run(self,scope,sources):
        started = time.monotonic()
        print('STAGE:',self.label,flush=True)
        for index in self.indices:
            exec(compile(sources[index],f'<credited-v22-cell-{index}>','exec'),scope)
        return dict(stage=self.label,seconds=round(time.monotonic()-started,2))

class MetricDistinctRunnerUp:
    def __init__(self, native_row):
        self.native_row = native_row
        self.considered = self.alternatives = 0

    def rank(self,tokens,engine,key,limit=25,placeholder='CCO'):
        original = self.native_row(tokens,engine,key,limit,placeholder)
        if original[1] != 'same_as_rank1':
            return original
        self.considered += 1
        engine = [engine] if isinstance(engine,str) else list(engine or [])
        top_key = key(tokens[0])
        # Inspect physical top3 only; metric twins do not consume the alternative hedge slot.
        for smile in engine[:3]:
            if not isinstance(smile,str) or any(c in smile for c in ';\\r\\n'):
                continue
            candidate = smile.strip()
            identity = key(candidate)
            if identity is not None and identity != top_key:
                self.alternatives += 1
                return self.native_row(tokens,[candidate],key,limit,placeholder)
        return original

class V22Experiment:
    def __init__(self,sources,runner_up=False):
        self.sources,self.runner_up = sources,runner_up
        self.scope = {'__name__':'__main__'}

    @staticmethod
    def check_bio(paths):
        assert paths, 'Missing sergiomalv/casmi-chebi-lmsd-structures (bio_meta.pkl)'
        assert all('casmi-chebi-lmsd-structures' in Path(p).parts for p in paths), (
            'Ambiguous BIO inputs: remove old prvsiyan/chebi-lipidmaps-casmi26; retain the V22 sergiomalv dataset',paths)

    def run(self):
        self.check_bio(glob.glob('/kaggle/input/**/bio_meta.pkl',recursive=True))
        scope = self.scope
        try:
            from IPython.display import display
        except ImportError:
            display = print
        scope['display'] = display
        stages = [V22Stage('Runtime, verified inputs and smoke switch',[3,5,6,8,9,11]),
            V22Stage('Engine2 and PubChem retrieval',[13,15]),
            V22Stage('Main engine and simulator evidence',[17,18,20,22]),
            V22Stage('rel150 promotion, fusion and official metric audit',[24,26,28,29,30,31]),
            V22Stage('Source fragment-gap, E006 and metric-exact rank2',[33,34,36])]
        timing = [stage.run(scope,self.sources) for stage in stages]
        assert scope['E2R2_STATS'] is not None, 'Source E2R2 did not complete'
        status = json.loads(Path('e2r2_stats.json').read_text())
        assert status.get('status')=='applied', 'Source E2R2 failed; do not submit a silently degraded recipe'
        control_bytes = Path('submission.csv').read_bytes()
        Path('submission_v22_control.csv').write_bytes(control_bytes)
        control = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        details = {}
        if self.runner_up:
            wrapper = MetricDistinctRunnerUp(scope['e2r2_row'])
            scope['e2r2_row'] = wrapper.rank
            workdir = Path('runner_up_work')
            workdir.mkdir(exist_ok=True)
            pre_hedge = Path('submission_base_e2r2.csv').read_bytes()
            (workdir/'submission.csv').write_bytes(pre_hedge)
            scope['_e2r2_apply'](scope,pre_hedge,workdir=str(workdir))
            treatment_bytes = (workdir/'submission.csv').read_bytes()
            treatment = pd.read_csv(workdir/'submission.csv',dtype=str,keep_default_na=False)
            assert treatment.columns.tolist() == control.columns.tolist()
            assert treatment.molecule_id.tolist() == control.molecule_id.tolist()
            assert (treatment.smiles.str.split(';').str[0] == control.smiles.str.split(';').str[0]).all()
            Path('submission.csv').write_bytes(treatment_bytes)
            details = dict(agreement_rows=wrapper.considered,runner_up_candidates=wrapper.alternatives)
        final = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        comparison = pd.DataFrame(dict(molecule_id=control.molecule_id,changed=final.smiles!=control.smiles))
        comparison.to_csv('v22_comparison.csv',index=False)
        report = dict(mode='metric_distinct_runner_up' if self.runner_up else 'v22_adaptation',
            measured_score=None,source_score_user_reported=.451,changed_rows_vs_v22=int(comparison.changed.sum()),
            control_sha256=hashlib.sha256(control_bytes).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),details=details,stages=timing)
        Path('v22_adaptation_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        if self.runner_up and not report['changed_rows_vs_v22']:
            print('NO-OP versus same-run V22: identical predictions do not merit a separate submission.')
        return report
'''


def build():
    raw = (SOURCE if SOURCE.exists() else UPLOAD).read_bytes()
    original = json.loads(raw)
    sources = {i:''.join(c['source']) for i,c in enumerate(original['cells']) if c['cell_type']=='code'}
    guard = "assert platform.python_version_tuple()[:2] == ('3', '12')"
    assert sources[9].count(guard)==1
    sources[9] = sources[9].replace(guard,"assert platform.python_version_tuple()[:2] in (('3', '12'), ('3', '13'))")
    header = 'CASMI26: Closing the Evidence Gap\nV22 — rel150, Updated BIO, Fragment Gap & Metric-Exact Hedge\n'
    notes = ('Compared with our confirmed 0.446 recipe this changes the BIO structure dataset, restores '
        'the v7c fragment-gap route and uses metric_identity rather than source score_key in the final '
        'engine2 rank2 hedge. These are multiple changes; no individual gain is established. '
        'The uploaded notebook has a successful Python3.12 12-molecule smoke run, but its E2R2 '
        'changed zero smoke rows. Public 0.451 is user-reported, not independently reproduced here. '
        'No score is claimed for this adaptation. Numerical kernels and source fallback remain, '
        'but a failed final E2R2 now prevents handing off a silently weaker submission. '
        'Our own orchestration is OOP; the credited source adapters remain substantially verbatim.')
    credits = ('Credits, attribution and limitations\nDirect source: user-uploaded '
        'casmi26-v22-rel150-e2r2-clean.ipynb; SHA256 '+hashlib.sha256(raw).hexdigest()+'. '
        'The file does not identify its uploader; provide its direct URL for a precise author link.\n'
        '@antriksharya — original rank2 hedge: https://www.kaggle.com/code/antriksharya/casmi26-lb-0-443-v7c-engine-2-rank-2-insert\n'
        'Omer Faruk Merey — metric-exact hedge, attributed by the uploaded V22 header; profile not verified.\n'
        '@seyitkaangunes — rel150 sweep, attributed by V22: https://www.kaggle.com/seyitkaangunes\n'
        '@sergiomalv — BIO structures: https://www.kaggle.com/datasets/sergiomalv/casmi-chebi-lmsd-structures\n'
        '@ahmedberatozer — engines/resources and v7c fragment-gap integration: https://www.kaggle.com/code/ahmedberatozer/casmi26-v7c-e006-fraggap\n'
        'Huseyin Emre Aksoy — reused fragment kernel: https://www.kaggle.com/huseyinemreaksoy\n'
        '@canqiang — E006; @samanyu1808 — ICE fallback; @flexonafft — metric fusion; '
        '@skomuro — promotion; @prvsiyan/@megayak — engine2 assets. Full inherited credits follow. '
        'Their wording is provenance, not verified performance/licence evidence. The inherited '
        'ahmedberatoze source line points to obstacledeveloper; use the explicit v7c link above. '
        'Retain upstream licences and ATTRIBUTION.md from BIO. Our contribution: orchestration, '
        'BIO ambiguity guard, run audit and optional runner-up policy, not the original models.')
    outputs = []
    for experiment in (False,True):
        metadata = {k:copy.deepcopy(original['metadata'][k]) for k in ('kernelspec','kaggle')}
        metadata['kaggle']['isGpuEnabled']=True
        metadata['casmi_experiment']=dict(source_sha256=hashlib.sha256(raw).hexdigest(),runner_up=experiment,
            measured_score=None,validation='Synthetic native E2R2 and wrapper checks; Kaggle full inference pending')
        method = ('Experimental hypothesis: when engine2 leader agrees with final rank1, hedge with '
            'the first different valid metric identity among its physical top3 at rank2. Otherwise '
            'preserve the native V22 row exactly. Rebuild from PRE-hedge E006 bytes, never stack '
            'two hedges. New risk: displacing a correct rank2 despite correct top1 consensus. '
            'No new models, no additional inference budget.') if experiment else 'Faithful numerical V22 recipe with runtime compatibility and explicit safety checks.'
        cells = [code_cell(prose(header+notes+'\n'+method)),
            code_cell(prose(credits)+prose(('Original V22 notes and inherited credits (quote delimiters normalized)\n'
                +''.join(original['cells'][0]['source'])+'\n'+''.join(original['cells'][4]['source'])).replace("'''",'’ ’ ’'))),
            code_cell(prose('Inputs and runtime\nNew mandatory BIO dataset relative to our older rel150: '
                'https://www.kaggle.com/datasets/sergiomalv/casmi-chebi-lmsd-structures . '
                'Remove prvsiyan/chebi-lipidmaps-casmi26 from these two notebooks: both expose '
                'bio_meta.pkl/bio_mass.npy/bio_fp.npy, and upstream discovery otherwise may select '
                'the wrong dataset. Preserve all other V22 input versions. The extra dataset metadata '
                'is retained, but Kaggle import can ignore attachment settings. Python3.13 additionally '
                'requires https://www.kaggle.com/datasets/eveliaveldrine/rdkit-wheels with RDKit2025.3.6 '
                'cp313; keep metric/rdkit-2026-3-3-wheel for the main process. GPU T4, Internet OFF. '
                'DreaMS is optional and not part of the V22 input manifest.')),
            code_cell(prose('OOP stages, source execution and paired audit\n'+method)+FRAMEWORK),
            code_cell(prose('Credited numerical kernels\nOnly the Python preflight guard is widened '
                'to 3.12/3.13. Scores, input discovery, fragment parameters, metric keys and hedge '
                'arithmetic are preserved. Full models and rendered outputs require Kaggle.')
                +'SOURCE_CELLS = '+repr(sources)+'\n'),
            code_cell(prose('Run and compare\nControl: submission_v22_control.csv. '
                'Reports: v22_adaptation_audit.json and v22_comparison.csv. Experimental hedge '
                'reports stay in runner_up_work. Smoke no-op does not prove hidden inactivity.')
                +f'experiment = V22Experiment(SOURCE_CELLS,runner_up={experiment!r})\nresult = experiment.run()\n')]
        nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=metadata)
        validate(nb)
        outputs.append(nb)
    return outputs


if __name__=='__main__':
    SOURCE.parent.mkdir(parents=True,exist_ok=True)
    if not SOURCE.exists():
        import shutil
        shutil.copy2(UPLOAD,SOURCE)
    for name,nb in zip(('casmi26_v22_adaptation.ipynb','casmi26_v22_metric_runner_up.ipynb'),build()):
        path=ROOT/'research/evidence-routing'/name
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
