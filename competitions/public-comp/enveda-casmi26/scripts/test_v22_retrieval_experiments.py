'''Synthetic tests for union rows/fingerprint alignment and pre-cutoff metric uniqueness.'''
import ast
import json
import pickle
import tempfile
import sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import numpy as np
from build_v22_retrieval_experiments import BIO_MODULE,UNIQUE_MODULE,build
from build_v22_adaptation import build as base_build

bio={};exec(BIO_MODULE,bio)
unique={};exec(UNIQUE_MODULE,unique)
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);new=root/'new';new.mkdir()
    with (new/'bio_meta.pkl').open('wb') as handle:
        pickle.dump(dict(keys=['A','B'],smiles=['newA','newB']),handle)
    np.save(new/'bio_fp.npy',np.array([[1,2],[3,4]],dtype=np.uint8))
    np.save(new/'bio_mass.npy',np.array([10.,20.]))
    with (root/'old.pkl').open('wb') as handle:
        pickle.dump(dict(keys=['A','C','C2','bad','D'],smiles=['oldA','oldC','oldC-twin','bad','oldD']),handle)
    calls=[]
    def fingerprint(smile):
        calls.append(smile)
        return (np.array([5,6],np.uint8),30.) if smile=='oldC' else None
    canonical=lambda s:dict(oldA='A',oldC='C',**{'oldC-twin':'C'},oldD='D').get(s)
    meta,fp,mass=bio['BioUnion'].load(new,root/'old.pkl',fingerprint,canonical,root/'report.json')
    assert meta==dict(keys=['A','B','C'],smiles=['newA','newB','oldC'])
    assert np.array_equal(fp,np.array([[1,2],[3,4],[5,6]],np.uint8))
    assert mass.tolist()==[10.,20.,30.] and calls==['oldC','oldD']
    report=json.loads((root/'report.json').read_text())
    assert report['added_rows']==1 and report['overlap_rows']==2 and report['invalid_rows']==2
    try:
        bio['BioUnion'].load(new,root/'old.pkl',lambda s:(np.array([1],np.uint8),30.),canonical,root/'bad.json')
    except AssertionError:
        pass
    else:
        raise AssertionError('Fingerprint shape mismatch must not be accepted')

identity=lambda s:None if s=='invalid' else s.split('-')[0]
selector=unique['UniqueShortlist'](identity)
# 80 distinct source keys collapse to one official identity. Backfill must scan past 80.
smiles=np.array(['A-'+str(i) for i in range(80)]+['B','C','D'],dtype=object)
record=dict(mid='m',cand=np.arange(len(smiles)))
result,audit=selector.select(record,np.arange(len(smiles),0,-1.),dict(smiles=smiles),lambda s:s,limit=3)
assert result['smiles']==['A-0','B','C'] and result['keys']==['A-0','B','C']
assert audit['native_count']==3 and audit['unique_count']==3 and audit['scanned_beyond_native80']
assert audit['metric_duplicates_skipped']==79 and audit['changed']
# Already metric-unique data preserves source order exactly, including stable score ties.
result,audit=selector.select(dict(mid='u',cand=np.arange(3)),np.ones(3),dict(smiles=np.array(['A','B','C'])),lambda s:s)
assert result['smiles']==['A','B','C'] and not audit['changed']
result,audit=selector.select(dict(mid='empty',cand=np.array([],dtype=int)),np.array([]),dict(smiles=[]),lambda s:s)
assert result==dict(smiles=[],keys=[]) and not audit['changed']

base=base_build()[0]
base_pack=ast.literal_eval(next(n.value for n in ast.parse(''.join(base['cells'][4]['source'])).body if isinstance(n,ast.Assign)))
for nb in build():
    pack=ast.literal_eval(next(n.value for n in ast.parse(''.join(nb['cells'][4]['source'])).body if isinstance(n,ast.Assign)))
    assert {i:s for i,s in pack.items() if i!=13}=={i:s for i,s in base_pack.items() if i!=13}
    tree=ast.parse(pack[13])
    files=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='ENG_FILES' for t in n.targets))
    for name,source in files.items():
        compile(source,name,'exec')
    runner=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='write' and n.args
        and isinstance(n.args[0],ast.Constant) and 'E.main(' in str(n.args[0].value))
    compile(runner,'engine-runner','exec')
    if nb['metadata']['casmi_experiment']['retrieval_mode']=='bio_union':
        assert 'bio_union_extension.py' in files and 'BioUnion.load' in files['casmi_engine.py']
        assert "if n in ('bio_fp.npy','bio_mass.npy','bio_meta.pkl')" in runner
    else:
        assert 'official_metric.py' in files and files['official_metric.py']==base_pack[28]
        assert 'selector.select(r,bl[mid],E.POOL,E.canon_key)' in runner
        # Execute the actual patched runner; mock only the expensive E.main/model boundary.
        with tempfile.TemporaryDirectory() as directory:
            extension=ModuleType('unique_shortlist');exec(UNIQUE_MODULE,extension.__dict__)
            metric=ModuleType('official_metric');metric.metric_identity=identity
            engine=ModuleType('casmi_engine')
            engine.find=lambda name:str(Path(directory)/name)
            engine.RANK=SimpleNamespace();engine.CFG=SimpleNamespace()
            engine.POOL=dict(smiles=np.array(['A-0','A-1','B','C'],dtype=object))
            engine.canon_key=lambda s:s
            records=[dict(mid='m',cand=np.arange(4))]
            scores={'m':np.array([4.,3.,2.,1.])}
            engine.main=lambda *args,**kwargs:({},records,dict(pv=scores,ours=scores))
            engine.blend_scores=lambda *args:scores
            out=Path(directory)/'out.json'
            with patch.dict(sys.modules,casmi_engine=engine,unique_shortlist=extension,official_metric=metric),patch.object(
                sys,'argv',['runner.py','test','train','sample',str(out)]):
                exec(compile(runner,'actual-runner','exec'),dict(__name__='__main__',__file__=str(Path(directory)/'eng_runner.py')))
            assert json.loads(out.read_text())['m']['smiles']==['A-0','B','C']
            assert json.loads((Path(directory)/'shortlist_audit.json').read_text())[0]['changed']
    for cell in nb['cells']:
        compile(''.join(cell['source']),'cell','exec')
        assert not cell['outputs'] and cell['execution_count'] is None
print('PASS: union overlap/invalid filtering, aligned regenerated fingerprints, incompatible width rejection, backfill beyond80, metric aliases/stable ties/empty cases and embedded runner compilation')
