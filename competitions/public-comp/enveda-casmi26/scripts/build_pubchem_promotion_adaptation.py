'''Attributed E006 promotion / rank-2 adaptations with explicit OOP orchestration.'''
import ast
import copy
import hashlib
import json
from pathlib import Path
from build_flow_experiments import ROOT, prose, validate
from build_metric_safe_adaptation import code_cell

UPLOAD = Path('/Users/flexonafft/Downloads/casmi26-e006-tuned-pubchem-promo-lb-0-444.ipynb')
SOURCE = ROOT/'archive/public-sources-2026-10-15/casmi26_e006_tuned_pubchem_promo_uploaded.ipynb'

FRAMEWORK = '''
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import pandas as pd

@dataclass(frozen=True)
class PromotionConfig:
    relative_threshold: float = 100.0

class CreditedStage:
    def __init__(self, label, indices):
        self.label, self.indices = label, indices

    def run(self, scope, sources):
        started = time.monotonic()
        print('STAGE:', self.label, flush=True)
        for index in self.indices:
            exec(compile(sources[index], f'<credited-promotion-cell-{index}>', 'exec'), scope)
        return dict(stage=self.label, seconds=round(time.monotonic()-started,2))

class EngineTwoHedge:
    def __init__(self, key):
        self.key = key

    def rank(self, tokens, engine):
        if not engine:
            return list(tokens)
        candidate = engine[0]
        identity = self.key(candidate)
        if identity is None or identity == self.key(tokens[0] if tokens else 'CCO'):
            return list(tokens)
        result, seen = [], set()
        for smile in tokens[:1]+[candidate]+tokens[1:]:
            key = self.key(smile) or smile
            if key not in seen:
                seen.add(key)
                result.append(smile)
        return result[:25]

class PromotionExperiment:
    def __init__(self, config, sources):
        self.config, self.sources = config, sources
        self.scope = {'__name__':'__main__'}

    def run(self):
        scope = self.scope
        try:
            from IPython.display import display
        except ImportError:
            display = print
        scope['display'] = display
        stages = [
            CreditedStage('Verified runtime and smoke switch',[8,10,13]),
            CreditedStage('Two-ranker retrieval and PubChem channel',[16,19]),
            CreditedStage('Main engine and pool-popularity ranking',[22,23,26]),
            CreditedStage('ICEBERG / GLACIER evidence',[29]),
            CreditedStage('PubChem promotion before fusion',[32,35]),
            CreditedStage('Official identity audit and fusion',[38,39,41,43]),
            CreditedStage('E006 ICE fallback; no fragment-gap',[45]),
            CreditedStage('Credited rank-2 hedge oracle',[47]),
        ]
        timing = [stage.run(scope,self.sources) for stage in stages]
        before = pd.read_csv('submission_e006.csv',dtype=str,keep_default_na=False)
        oracle = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        hedge = EngineTwoHedge(scope['_keyp'])
        rows = []
        for row in before.itertuples(index=False):
            tokens = [s for s in row.smiles.split(';') if s]
            engine = scope['ENG'].get(row.molecule_id,{})
            candidates = [s for s in (engine.get('smiles') or []) if s and s != 'CCO']
            revised = hedge.rank(tokens,candidates)
            rows.append((row.molecule_id,';'.join(revised)))
        frame = pd.DataFrame(rows,columns=['molecule_id','smiles'])
        assert frame.equals(oracle), 'OOP/source rank-2 parity failed'
        assert frame.molecule_id.is_unique and list(frame.molecule_id) == list(before.molecule_id)
        assert frame.smiles.map(lambda s:1 <= len(s.split(';')) <= 25).all()
        changed = before.smiles != frame.smiles
        assert (before.smiles.str.split(';').str[0] == frame.smiles.str.split(';').str[0]).all()
        report = dict(relative_threshold=self.config.relative_threshold,
            rank2_changed_rows=int(changed.sum()),rank2_changed_top1=0,source_rank2_parity=True,
            measured_score=None,stages=timing,submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
            note='Rank2 changes compare against same-run E006, not the old 0.441 submission. Promotion can change top1 upstream.')
        Path('promotion_adaptation_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return report
'''


def build():
    origin = SOURCE if SOURCE.exists() else UPLOAD
    raw = origin.read_bytes()
    original = json.loads(raw)
    sources = {i:''.join(c['source']) for i,c in enumerate(original['cells']) if c['cell_type']=='code'}
    guard = "assert platform.python_version_tuple()[:2] == ('3', '12')"
    assert sources[10].count(guard) == 1
    sources[10] = sources[10].replace(guard,"assert platform.python_version_tuple()[:2] in (('3', '12'), ('3', '13'))")
    assert "MODE = 'rank2'" in sources[47]
    assert not any('FRAG_GAP' in s for s in sources.values())
    outputs = []
    for threshold in (100.,150.):
        pack = dict(sources)
        assert pack[32].count('(rel > 100.0)') == 1
        pack[32] = pack[32].replace('(rel > 100.0)',f'(rel > {threshold})')
        # Keep inherited comments accurate for the isolated threshold experiment.
        pack[32] = pack[32].replace('rel promotion threshold 100.0',f'rel promotion threshold {threshold}')
        metadata = copy.deepcopy(original['metadata'])
        metadata['kaggle']['isGpuEnabled'] = True
        metadata['casmi_experiment'] = dict(relative_threshold=threshold,measured_score=None,
            source_sha256=hashlib.sha256(raw).hexdigest(),runtime_compatibility='Python 3.12/3.13 with compatible simulator wheels',
            validation='Source arithmetic retained; synthetic hedge parity; full Kaggle execution pending')
        cells = [code_cell(prose('CASMI26: Closing the Evidence Gap\nPubChem Promotion + Engine-2 Hedge\n'
            +f'Variant: rel > {threshold:g}. '+('Adaptation of the uploaded recipe.' if threshold==100 else 'Single-factor threshold experiment versus rel100.')
            +'\nThis is NOT our fragment-gap 0.441 baseline with one edited line. It reproduces E006 '
            'without fragment-gap, promotes PubChem before metric fusion, then applies the source rank-2 hedge. '
            'No scores are independently reproduced yet. The uploaded author reports 0.444 at 100 and '
            '0.446 at 150; public-threshold selection may overfit. Full inference and visual outputs '
            'are unverified locally. OOP adapters retain substantial upstream procedural code verbatim.')),
            code_cell(prose('Credits and contribution boundaries\n'
                'Direct source: user-uploaded casmi26-e006-tuned-pubchem-promo-lb-0-444.ipynb; SHA256 '
                +hashlib.sha256(raw).hexdigest()+'. Its uploader is not identified in the supplied metadata; '
                'threshold-sweep authorship belongs to that source author, not to this adaptation. '
                'Please provide the direct notebook URL to add an exact author link.\n'
                '@antriksharya — E006 + engine-2 rank-2 hedge: https://www.kaggle.com/code/antriksharya/casmi26-e006-e2rank2\n'
                '@skomuro — popularity OR structure-evidence promotion: https://www.kaggle.com/code/skomuro/casmi26-silent-failures-in-the-public-pipeline\n'
                'Our contribution: OOP stage organization, runtime compatibility, hedge parity and run audit. '
                'Appearance/OOP does not transfer authorship. All original component credits and licence caveats follow.')
                +prose('Original source explanation and full credits (verbatim)\n'+''.join(original['cells'][0]['source']))
                +sources[3]),
            code_cell(prose('Inputs and runtime\n'
                'Preserve the uploaded dataset versions in notebook metadata; import may not attach them automatically. '
                'The source attaches DreaMS as optional tail fill; preserve it for the closest reproduction. '
                'Python3.13 requires eveliaveldrine/rdkit-wheels containing RDKit2025.3.6 cp313; '
                'keep metric/rdkit-2026-3-3-wheel for the main runtime. GPU T4, Internet OFF. '
                'Only the Python preflight guard is widened; original version/platform wheel selection remains intact. '
                'An inherited reference below to Python3.12 describes the original run, not the widened adapter.')),
            code_cell(prose('Object-oriented orchestration and source-parity checker\n'
                'Execute the unchanged numerical source stages in an explicit scope. Retain the original '
                'rank2 result as an oracle and independently reconstruct it with EngineTwoHedge.')+FRAMEWORK),
            code_cell(prose('Credited source pack\n'
                'Only runtime acceptance and the declared promotion threshold differ. No new trained models, '
                'no previous failed fragment experiments, no post-fusion forced top1 promotion.')+'SOURCE_CELLS = '+repr(pack)+'\n'),
            code_cell(prose('Run and inspect\n'
                'promotion_adaptation_audit.json describes current execution. Source audit files remain intact. '
                'submission_e006.csv is the same-run pre-hedge control. A smoke result is not validation accuracy.')
                +f'config = PromotionConfig(relative_threshold={threshold})\n'
                +'experiment = PromotionExperiment(config,SOURCE_CELLS)\nresult = experiment.run()\n')]
        nb = dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=metadata)
        validate(nb)
        outputs.append(nb)
    return outputs


if __name__=='__main__':
    SOURCE.parent.mkdir(parents=True,exist_ok=True)
    if not SOURCE.exists():
        import shutil
        shutil.copy2(UPLOAD,SOURCE)
    for threshold,nb in zip((100,150),build()):
        path = ROOT/f'research/evidence-routing/casmi26_pubchem_promotion_rel{threshold}.ipynb'
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
