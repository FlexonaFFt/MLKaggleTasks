'''Independent BIO coverage and early metric-unique retrieval experiments on V22.'''
import ast
import copy
import json
from build_v22_adaptation import build as base_build,FRAMEWORK
from build_flow_experiments import ROOT,prose,validate
from build_metric_safe_adaptation import code_cell

BIO_MODULE = '''
import json
import pickle
import time
from pathlib import Path
import numpy as np

class BioUnion:
    @staticmethod
    def load(new_dir,old_meta_path,fingerprint,canonical_key,report_path):
        new_dir=Path(new_dir)
        with (new_dir/'bio_meta.pkl').open('rb') as handle:
            new=pickle.load(handle)
        with Path(old_meta_path).open('rb') as handle:
            old=pickle.load(handle)
        fp=np.load(new_dir/'bio_fp.npy')
        mass=np.load(new_dir/'bio_mass.npy')
        assert fp.ndim==2 and fp.dtype==np.uint8
        assert len(new['keys'])==len(new['smiles'])==len(fp)==len(mass)
        assert np.isfinite(mass).all() and (mass>0).all()
        assert len(old['keys'])==len(old['smiles'])
        seen=set(new['keys'])
        extra_keys,extra_smiles,extra_fp,extra_mass=[],[],[],[]
        invalid=overlap=0
        started=time.monotonic()
        for old_key,smile in zip(old['keys'],old['smiles']):
            if old_key in seen:
                overlap+=1
                continue
            # Never silently change the union into a time-dependent partial dataset.
            if time.monotonic()-started>1200:
                raise TimeoutError('BIO union fingerprint preparation exceeded 1200s; no partial union accepted')
            key=canonical_key(smile)
            if key is None:
                invalid+=1
                continue
            if key in seen:
                overlap+=1
                continue
            result=fingerprint(smile)
            if result is None:
                invalid+=1
                continue
            bits,m=result
            bits=np.asarray(bits)
            assert bits.ndim==1 and bits.shape[0]==fp.shape[1] and bits.dtype==fp.dtype, 'Incompatible recalculated fingerprint'
            if not np.isfinite(m) or m<=0:
                invalid+=1
                continue
            seen.add(key)
            extra_keys.append(key);extra_smiles.append(smile);extra_fp.append(bits);extra_mass.append(float(m))
        meta=dict(keys=list(new['keys'])+extra_keys,smiles=list(new['smiles'])+extra_smiles)
        if extra_keys:
            fp=np.vstack([fp,np.stack(extra_fp)])
            mass=np.concatenate([mass,np.asarray(extra_mass,dtype=mass.dtype)])
        report=dict(new_rows=len(new['keys']),old_rows=len(old['keys']),added_rows=len(extra_keys),
            overlap_rows=overlap,invalid_rows=invalid,seconds=round(time.monotonic()-started,2),
            new_directory=str(new_dir),old_meta_path=str(old_meta_path),
            key_policy='source engine2 canon_key; preserve new dataset priority',
            fingerprint_policy='recalculate every added structure with engine2 fp_and_mass; never reuse old fingerprint arrays')
        Path(report_path).write_text(json.dumps(report,indent=2))
        print('BIO_UNION',json.dumps(report),flush=True)
        return meta,fp,mass
'''

UNIQUE_MODULE = '''
import json
from pathlib import Path
import numpy as np

class UniqueShortlist:
    def __init__(self,identity):
        self.identity=identity

    def select(self,record,scores,pool,source_key,limit=40):
        order=np.argsort(-scores,kind='mergesort')
        native,native_seen=[],set()
        for c in record['cand'][order[:80]]:
            smile=pool['smiles'][c];key=source_key(smile)
            if key is not None and key not in native_seen:
                native_seen.add(key);native.append(smile)
                if len(native)>=limit:
                    break
        smiles,keys,seen=[],[],set()
        examined=duplicates=0
        for position in order:
            examined+=1
            smile=pool['smiles'][record['cand'][position]]
            key=source_key(smile)
            if key is None:
                continue
            metric=self.identity(smile)
            if metric is None:
                continue
            if metric in seen:
                duplicates+=1
                continue
            seen.add(metric);smiles.append(smile);keys.append(key)
            if len(smiles)>=limit:
                break
        return dict(smiles=smiles,keys=keys),dict(molecule_id=str(record['mid']),
            native_count=len(native),unique_count=len(smiles),examined=examined,
            metric_duplicates_skipped=duplicates,changed=smiles!=native,
            scanned_beyond_native80=examined>80)
'''


def patch_engine(source,mode,metric_source):
    tree=ast.parse(source)
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENG_FILES' for t in n.targets))
    files=ast.literal_eval(assignment.value)
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='write' and n.args and isinstance(n.args[0],ast.Constant)
        and 'order = np.argsort(-bl[mid]' in str(n.args[0].value))
    runner=ast.literal_eval(call.args[0])
    if mode=='bio_union':
        files['bio_union_extension.py']=BIO_MODULE
        engine=files['casmi_engine.py']
        old='    bd = os.path.dirname(find("bio_fp.npy")); bm = pickle.load(open(os.path.join(bd, "bio_meta.pkl"), "rb"))'
        new='''    bd = os.path.dirname(find("bio_fp.npy"))
    from bio_union_extension import BioUnion
    old_paths=sorted(glob.glob('/kaggle/input/**/chebi-lipidmaps-casmi26/**/bio_meta.pkl',recursive=True))
    assert len(old_paths)==1, ('Attach exactly one prvsiyan/chebi-lipidmaps-casmi26 version',old_paths)
    _fp_init()
    bm,bio_fp,bio_mass=BioUnion.load(bd,old_paths[0],fp_and_mass,canon_key,
        os.path.join(os.path.dirname(__file__),'bio_union_audit.json'))'''
        assert engine.count(old)==1
        engine=engine.replace(old,new)
        engine=engine.replace('np.load(os.path.join(bd, "bio_fp.npy"))[bnew]','bio_fp[bnew]')
        engine=engine.replace('np.load(os.path.join(bd, "bio_mass.npy"))[bnew]','bio_mass[bnew]')
        files['casmi_engine.py']=engine
        anchor="        if n == 'fp_bits.npy':"
        assert runner.count(anchor)==1
        runner=runner.replace(anchor,"""        if n in ('bio_fp.npy','bio_mass.npy','bio_meta.pkl'):
            import glob
            hits=sorted(glob.glob('/kaggle/input/**/casmi-chebi-lmsd-structures/**/'+n,recursive=True))
            assert len(hits)==1, ('BIO path must be pinned to sergiomalv',n,hits)
            return hits[0]
"""+anchor)
    else:
        files['unique_shortlist.py']=UNIQUE_MODULE
        files['official_metric.py']=metric_source
        start=runner.index('        order = np.argsort(-bl[mid]')
        end=runner.index('        res[str(mid)] = dict(smiles=smis, keys=keys)')+len('        res[str(mid)] = dict(smiles=smis, keys=keys)')
        runner=runner[:start]+'''        result,audit=selector.select(r,bl[mid],E.POOL,E.canon_key)
        res[str(mid)]=result
        shortlist_audit.append(audit)'''+runner[end:]
        runner=runner.replace('    res = {}\n','''    from unique_shortlist import UniqueShortlist
    from official_metric import metric_identity
    selector=UniqueShortlist(metric_identity)
    shortlist_audit=[]
    res = {}
''')
        runner=runner.replace("    json.dump(res, open(out, 'w'))",'''    json.dump(shortlist_audit,open(os.path.join(here,'shortlist_audit.json'),'w'))
    print('METRIC_UNIQUE_SHORTLIST',sum(a['changed'] for a in shortlist_audit),'changed molecules',flush=True)
    json.dump(res, open(out, 'w'))''')
    return source.replace(ast.get_source_segment(source,assignment),'ENG_FILES = '+repr(files)).replace(
        ast.get_source_segment(source,call.args[0]),repr(runner))


def build():
    base=base_build()[0]
    assignment=next(n for n in ast.parse(''.join(base['cells'][4]['source'])).body if isinstance(n,ast.Assign))
    original=ast.literal_eval(assignment.value)
    outputs=[]
    for mode,title,method in (
        ('bio_union','V22 — BIO Structure Union',
         'Keep the sergiomalv BIO entries first and add old prvsiyan BIO structures absent by engine2 source key. '
         'Recalculate added fingerprints and masses with engine2 native fp_and_mass and the same selected bits. '
         'No assumptions about old fingerprint-bit ordering. Source-key overlaps keep the new version; '
         'full official metric deduplication is NOT introduced into the pool. Candidate discovery is pinned '
         'to sergiomalv; the old dataset is read only by the extension. A 1200s preparation deadline raises '
         'rather than silently accepting partial coverage. More candidates may reduce accuracy or exceed runtime.'),
        ('unique_backfill','V22 — Early Metric-Unique Backfill',
         'Keep the original BIO and engine2 candidate scoring, but export up to 40 official metric-unique '
         'candidates before downstream fusion/simulators. Scan stable score order until 40 unique identities '
         'or exhaustion, instead of stopping inspection at 80 source-ranked candidates. '
         'Keep original source keys in the exported keys field for downstream compatibility; use official '
         'metric_identity only for uniqueness. No top40-to-top60 expansion and no changed scoring weights. '
         'More unique candidates can change simulator inputs and increase canonicalization runtime.')):
        nb=copy.deepcopy(base)
        pack=dict(original)
        pack[13]=patch_engine(pack[13],mode,pack[28])
        framework=FRAMEWORK
        if mode=='bio_union':
            before="        self.check_bio(glob.glob('/kaggle/input/**/bio_meta.pkl',recursive=True))"
            after="""        bio=glob.glob('/kaggle/input/**/bio_meta.pkl',recursive=True)
        assert any('casmi-chebi-lmsd-structures' in Path(p).parts for p in bio), 'Attach sergiomalv BIO'
        assert any('chebi-lipidmaps-casmi26' in Path(p).parts for p in bio), 'Attach prvsiyan BIO for union'
        assert all(any(s in Path(p).parts for s in ('casmi-chebi-lmsd-structures','chebi-lipidmaps-casmi26')) for p in bio), ('Unexpected BIO input',bio)"""
            assert framework.count(before)==1
            framework=framework.replace(before,after)
        # Parent report compares source and final treatment, NOT a separately run 0.451 recipe.
        framework=framework.replace("mode='metric_distinct_runner_up' if self.runner_up else 'v22_adaptation'",f"mode={mode!r}")
        framework=framework.replace('changed_rows_vs_v22','changed_rows_vs_same_run_source_output')
        nb['cells'][0]=code_cell(prose('CASMI26: Closing the Evidence Gap\n'+title+'\n'
            'Independent retrieval experiment on the user-confirmed V22 0.451 recipe. Treatment score unknown. '
            +method+'\nAll rel150, fragment-gap, fusion and metric-exact rank2 policies stay unchanged. '
            'Do not combine with prior runner-up/rel200/fragment ablations. No same-run unmodified V22 '
            'neural control is computed: changes begin inside retrieval and a second full inference is avoided. '
            'Source parity compares the adapted recipe to its own output, NOT predictions from 0.451.'))
        input_note=('BIO union requires BOTH https://www.kaggle.com/datasets/sergiomalv/casmi-chebi-lmsd-structures '
            'AND https://www.kaggle.com/datasets/prvsiyan/chebi-lipidmaps-casmi26 . This overrides the '
            'single-BIO advice in the original source notes retained in Credits. The extra old input '
            'must be attached manually; do not assume notebook metadata adds it.') if mode=='bio_union' else (
            'No new datasets. Keep only sergiomalv BIO as in successful V22; do not attach the old BIO here.')
        nb['cells'][2]=code_cell(prose('Inputs and runtime\n'+input_note+'\n'
            'Keep all other successful V22 inputs and compatible RDKit wheels. Python3.13 requires '
            'eveliaveldrine/rdkit-wheels with RDKit2025.3.6 cp313; main runtime uses RDKit2026.3.3. '
            'GPU T4, Internet OFF. Full data access, fingerprint generation, GPU runtime and rendered '
            'outputs are not validated locally. Original source code/data licences still apply.'))
        nb['cells'][1]['source']+=prose('\nAdditional experiment attribution\n'
            '@sergiomalv — new BIO structures; @prvsiyan — old BIO candidates and fingerprint code. '
            'Our contribution: explicit BIO union/recalculated additions or early metric-unique export, '
            'not the original engines, metric definition or candidate databases. '+method).splitlines(keepends=True)
        nb['cells'][3]=code_cell(prose('OOP orchestration\n'+method)+framework)
        nb['cells'][4]=code_cell(prose('Credited source pack with one engine2 extension\n'
            'Only engine2 retrieval/export changes, plus the union input guard when needed. '
            'All other numerical cells retain V22 arithmetic.')+'SOURCE_CELLS = '+repr(pack)+'\n')
        nb['cells'][5]=code_cell(prose('Run and inspect\n'
            'The retrieval report is eng/bio_union_audit.json or eng/shortlist_audit.json. '
            'submission_v22_control.csv here is the adapted source output, not an unmodified 0.451 '
            'control. v22_adaptation_audit.json reports this explicitly. Inspect actual additions/list '
            'changes before spending a submission; score gains are unverified.')
            +'experiment = V22Experiment(SOURCE_CELLS,runner_up=False)\nresult = experiment.run()\n')
        nb['metadata']['casmi_experiment'].update(retrieval_mode=mode,baseline_user_confirmed_score=.451,
            validation='Synthetic pool union and shortlist tests; full Kaggle data/GPU execution pending')
        validate(nb);outputs.append(nb)
    return outputs


if __name__=='__main__':
    for name,nb in zip(('casmi26_v22_bio_union.ipynb','casmi26_v22_unique_backfill.ipynb'),build()):
        path=ROOT/'research/evidence-routing'/name
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
