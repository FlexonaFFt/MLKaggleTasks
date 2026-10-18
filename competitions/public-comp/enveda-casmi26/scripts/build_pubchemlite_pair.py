'''Public E006+PubChemLite recipe and independent V22+PubChemLite coverage experiment.'''
import ast
import copy
import hashlib
import json
from pathlib import Path
from build_pubchem_promotion_adaptation import FRAMEWORK as E006_FRAMEWORK
from build_v22_adaptation import build as v22_build
from build_flow_experiments import ROOT,prose,validate
from build_metric_safe_adaptation import code_cell

UPLOAD=Path('/Users/flexonafft/Downloads/casmi26-v459-e006-defaultimg.ipynb')
SOURCE=ROOT/'archive/public-sources-2026-10-18/casmi26_v459_uploaded.ipynb'

INPUTS='''
class PubChemLiteInputs:
    @staticmethod
    def check(bio_paths,lite_paths,bio_slug):
        assert len(bio_paths)==1 and bio_slug in Path(bio_paths[0]).parts, (
            'Attach only the BIO dataset selected for this recipe: '+bio_slug,bio_paths)
        assert len(lite_paths)==1, ('Attach franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints',lite_paths)
        folder=Path(lite_paths[0]).parent
        for name in ('pubchemlite_fp.npy','pubchemlite_mass.npy','pubchemlite_meta.pkl'):
            assert (folder/name).is_file(), 'Incomplete PubChemLite input: '+str(folder/name)

    @classmethod
    def verify(cls,bio_slug):
        import glob
        cls.check(glob.glob('/kaggle/input/**/bio_meta.pkl',recursive=True),
            glob.glob('/kaggle/input/**/pubchemlite_meta.pkl',recursive=True),bio_slug)
'''


def engine_files(source):
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='ENG_FILES' for t in n.targets))
    return assignment,ast.literal_eval(assignment.value)


def pubchemlite_block(source):
    _,files=engine_files(source)
    engine=files['casmi_engine.py']
    start=engine.index('    # ADDED (Hongbin Gu): PubChemLite CC0')
    end=engine.index('    tr = pd.DataFrame',start)
    block=engine[start:end]
    assert 'pubchemlite_fp.npy' in block and 'pubchemlite_meta.pkl' in block
    return block


def add_pubchemlite(source,block):
    assignment,files=engine_files(source)
    engine=files['casmi_engine.py']
    assert 'pubchemlite_fp.npy' not in engine
    anchor='    tr = pd.DataFrame({"ik": L["ik"], "smi": L["smi"]}).dropna().drop_duplicates("ik")'
    assert engine.count(anchor)==1
    files['casmi_engine.py']=engine.replace(anchor,block+anchor)
    return source.replace(ast.get_source_segment(source,assignment),'ENG_FILES = '+repr(files))


def build():
    raw=(SOURCE if SOURCE.exists() else UPLOAD).read_bytes()
    original=json.loads(raw)
    pack={i:''.join(c['source']) for i,c in enumerate(original['cells']) if c['cell_type']=='code'}
    assert '(rel > 150.0)' in pack[32] and "MODE = 'rank2'" in pack[47]
    assert not any('FRAG_GAP' in s for s in pack.values())
    block=pubchemlite_block(pack[16])
    provenance=('Direct snapshot: casmi26-v459-e006-defaultimg.ipynb; SHA256 '+hashlib.sha256(raw).hexdigest()+'. '
        'Uploader identity/direct notebook URL is absent from the supplied file. Public 0.454 is '
        'reported by the user; filename v459 and inherited markdown 0.446 are NOT independent score evidence. '
        'Hongbin Gu — PubChemLite engine2 pool insertion, explicitly credited in the source code. '
        '@franciscoangulo — candidate fingerprints: https://www.kaggle.com/datasets/franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints . '
        'The resource is labelled CC0 by the source; original component licences and dataset terms '
        'remain authoritative. Preserve its own attribution. Our contribution is OOP adaptation, '
        'input guards and the controlled V22 integration, not PubChemLite or the original engines. '
        'All inherited author credits and links follow; OOP/presentation does not transfer authorship.')
    runtime=('Python3.13 successful source smoke: RDKit2026.03.3 main environment and isolated '
        'RDKit2025.3.6 cp313 simulator wheel. Keep metric/rdkit-2026-3-3-wheel AND '
        'eveliaveldrine/rdkit-wheels. GPU T4, Internet OFF. Import may not attach inputs automatically. '
        'PubChemLite filenames are checked before neural inference. Real fingerprint compatibility, '
        'full GPU inference, hidden runtime and rendered outputs remain unverified locally.')
    metadata={k:copy.deepcopy(original['metadata'][k]) for k in ('kernelspec','kaggle')}
    metadata['kaggle']['isGpuEnabled']=True
    metadata['casmi_experiment']=dict(source_sha256=hashlib.sha256(raw).hexdigest(),mode='e006_pubchemlite_adaptation',
        relative_threshold=150.,measured_score=None,validation='Synthetic native pool-extension tests; Kaggle full inference pending')
    native=dict(nbformat=4,nbformat_minor=5,metadata=metadata,cells=[
        code_cell(prose('CASMI26: Closing the Evidence Gap\nE006 + PubChemLite — Source Adaptation\n'
            'Preserve the uploaded recipe: old prvsiyan BIO, rel150 promotion, no fragment-gap, '
            'ordinary source-key rank2 hedge, PubChemLite in engine2. Do not silently substitute V22 '
            'components. Source score is not yet reproduced. Multiple components differ from V22 0.451.')),
        code_cell(prose('Credits and source lineage\n'+provenance)+prose(
            ('Original explanation and full credits (historical claims/settings, not current guarantees)\n'
             +''.join(original['cells'][0]['source'])).replace("'''",'’ ’ ’'))+pack[3]),
        code_cell(prose('Inputs\nBIO: prvsiyan/chebi-lipidmaps-casmi26 ONLY. Disconnect sergiomalv BIO '
            'in this notebook. Add franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints. '
            'Keep the remaining inputs/versions of the uploaded recipe. Optional DreaMS remains '
            'attached in its metadata; we do not add a new DreaMS algorithm.\n'+runtime)),
        code_cell(prose('Object-oriented orchestration and source-key hedge parity\n'
            'Substantial numerical source cells are retained verbatim; source hedge is checked '
            'against its OOP implementation. This is not a rewrite of the predictors.')+E006_FRAMEWORK+INPUTS),
        code_cell(prose('Credited numerical source pack\nAll executable code is preserved from '
            'the upload, including default-image preflight changes. No hidden score is inferred '
            'from the visible placeholder smoke.')+'SOURCE_CELLS = '+repr(pack)+'\n'),
        code_cell(prose('Run\nThe hedge audit compares against same-run PRE-hedge E006, NOT the '
            'previous no-PubChemLite baseline. PubChemLite pool additions are printed by engine2. '
            'This run does not compute a second complete neural control without PubChemLite.')
            +"PubChemLiteInputs.verify('chebi-lipidmaps-casmi26')\n"
            +'config = PromotionConfig(relative_threshold=150.)\n'
            +'experiment = PromotionExperiment(config,SOURCE_CELLS)\nresult = experiment.run()\n')])
    hybrid=copy.deepcopy(v22_build()[0])
    node=next(n for n in ast.parse(''.join(hybrid['cells'][4]['source'])).body if isinstance(n,ast.Assign))
    hybrid_pack=ast.literal_eval(node.value)
    hybrid_pack[13]=add_pubchemlite(hybrid_pack[13],block)
    hybrid['cells'][0]=code_cell(prose('CASMI26: Closing the Evidence Gap\nV22 + PubChemLite — Coverage Experiment\n'
        'Single candidate-source addition to the user-confirmed V22 0.451 recipe. Keep sergiomalv BIO, '
        'rel150, fragment-gap and metric-exact engine2 rank2. Add only the credited PubChemLite '
        'engine2 pool block from the new source; do not add BIO union, early unique backfill or '
        'runner-up hedge. More candidates can improve recall but also increase competition, memory '
        'and runtime. No score improvement is claimed.'))
    hybrid['cells'][1]['source']+=prose('PubChemLite integration provenance\n'+provenance).splitlines(keepends=True)
    hybrid['cells'][2]=code_cell(prose('Inputs\nKeep sergiomalv/casmi-chebi-lmsd-structures ONLY '
        'as BIO, as in successful V22. Disconnect old prvsiyan BIO in THIS notebook. Additionally '
        'attach franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints MANUALLY; it is not '
        'present in inherited V22 metadata. All other V22 inputs remain.\n'+runtime))
    hybrid['cells'][3]['source']+=INPUTS.splitlines(keepends=True)
    hybrid['cells'][4]=code_cell(prose('Credited V22 numerical kernels plus PubChemLite pool insertion\n'
        'Only the embedded engine2 build_pool gains the exact upstream insertion block. Other '
        'V22 source cells, ranking weights, cutoffs and hedge identity remain unchanged.')
        +'SOURCE_CELLS = '+repr(hybrid_pack)+'\n')
    hybrid['cells'][5]=code_cell(prose('Run\nThis is a retrieval change requiring full inference. '
        'submission_v22_control.csv is the SAME-RUN modified recipe output, not an unmodified '
        '0.451 control. No independent baseline predictions are computed. The engine2 log reports '
        'actual PubChemLite additions; a completed smoke run does not establish hidden benefit.')
        +"PubChemLiteInputs.verify('casmi-chebi-lmsd-structures')\n"
        +'experiment = V22Experiment(SOURCE_CELLS,runner_up=False)\nresult = experiment.run()\n')
    hybrid['metadata']['casmi_experiment'].update(mode='v22_pubchemlite',baseline_user_confirmed_score=.451,
        pubchemlite_source_sha256=hashlib.sha256(raw).hexdigest(),measured_score=None)
    # Avoid labelling a modified retrieval run as a paired comparison with the original 0.451.
    framework=''.join(hybrid['cells'][3]['source']).replace('changed_rows_vs_v22','changed_rows_vs_same_run_source_output')
    framework=framework.replace("mode='metric_distinct_runner_up' if self.runner_up else 'v22_adaptation'","mode='v22_pubchemlite'")
    hybrid['cells'][3]=code_cell(framework)
    for nb in (native,hybrid):
        validate(nb)
    return native,hybrid


if __name__=='__main__':
    SOURCE.parent.mkdir(parents=True,exist_ok=True)
    if not SOURCE.exists():
        import shutil
        shutil.copy2(UPLOAD,SOURCE)
    for name,nb in zip(('casmi26_e006_pubchemlite_adaptation.ipynb','casmi26_v22_pubchemlite.ipynb'),build()):
        path=ROOT/'research/evidence-routing'/name
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
