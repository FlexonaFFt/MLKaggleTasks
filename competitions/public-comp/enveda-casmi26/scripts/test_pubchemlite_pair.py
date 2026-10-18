'''Exact source parity and real PubChemLite insertion block on synthetic aligned arrays.'''
import ast
import json
import os
import pickle
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from build_pubchemlite_pair import SOURCE,UPLOAD,INPUTS,build,engine_files,pubchemlite_block
from build_v22_adaptation import build as v22_build

original=json.loads((SOURCE if SOURCE.exists() else UPLOAD).read_bytes())
source={i:''.join(c['source']) for i,c in enumerate(original['cells']) if c['cell_type']=='code'}
block=pubchemlite_block(source[16])
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    with (root/'pubchemlite_meta.pkl').open('wb') as handle:
        pickle.dump(dict(keys=['A','C','D'],smiles=['duplicateA','newC','newD']),handle)
    np.save(root/'pubchemlite_fp.npy',np.array([[9,9],[5,6],[7,8]],np.uint8))
    np.save(root/'pubchemlite_mass.npy',np.array([99.,30.,40.]))
    env=dict(os=os,np=np,pd=pd,pickle=pickle,find=lambda name:str(root/name),
        co_keys=np.array(['A','B'],object),co_smi=np.array(['oldA','oldB'],object),
        co_fp=np.array([[1,2],[3,4]],np.uint8),co_mass=np.array([10.,20.]),log=lambda message:None)
    exec(block if not block.startswith('    ') else '\n'.join(l[4:] for l in block.splitlines()),env)
    assert env['co_keys'].tolist()==['A','B','C','D']
    assert env['co_smi'].tolist()==['oldA','oldB','newC','newD']
    assert env['co_mass'].tolist()==[10.,20.,30.,40.]
    assert np.array_equal(env['co_fp'],np.array([[1,2],[3,4],[5,6],[7,8]],np.uint8))
    checks={'Path':Path};exec(INPUTS,checks)
    checks['PubChemLiteInputs'].check(['/x/chebi-lipidmaps-casmi26/bio_meta.pkl'],[str(root/'pubchemlite_meta.pkl')],'chebi-lipidmaps-casmi26')
    for paths in ([],['/x/casmi-chebi-lmsd-structures/bio_meta.pkl'],
        ['/x/chebi-lipidmaps-casmi26/bio_meta.pkl','/x/casmi-chebi-lmsd-structures/bio_meta.pkl']):
        try:
            checks['PubChemLiteInputs'].check(paths,[str(root/'pubchemlite_meta.pkl')],'chebi-lipidmaps-casmi26')
        except AssertionError:
            pass
        else:
            raise AssertionError('Wrong or ambiguous BIO was accepted')
    (root/'pubchemlite_mass.npy').unlink()
    try:
        checks['PubChemLiteInputs'].check(['/x/chebi-lipidmaps-casmi26/bio_meta.pkl'],[str(root/'pubchemlite_meta.pkl')],'chebi-lipidmaps-casmi26')
    except AssertionError:
        pass
    else:
        raise AssertionError('Incomplete PubChemLite was accepted')

def extract(nb):
    return ast.literal_eval(next(n.value for n in ast.parse(''.join(nb['cells'][4]['source'])).body if isinstance(n,ast.Assign)))

native,hybrid=build()
assert extract(native)==source, 'Native computational code must remain exact'
assert native['metadata']['kaggle']['dataSources']==original['metadata']['kaggle']['dataSources']
baseline=v22_build()[0]
base_pack=extract(baseline);hybrid_pack=extract(hybrid)
assert {i:s for i,s in hybrid_pack.items() if i!=13}=={i:s for i,s in base_pack.items() if i!=13}
_,base_files=engine_files(base_pack[13]);_,hybrid_files=engine_files(hybrid_pack[13])
for name,s in hybrid_files.items():
    assert s.replace(block,'')==base_files[name] if name=='casmi_engine.py' else s==base_files[name]
    compile(s,name,'exec')
for nb in (native,hybrid):
    for i,s in extract(nb).items():
        compile(s,f'source-{i}','exec')
    for c in nb['cells']:
        compile(''.join(c['source']),'cell','exec')
        assert not c['outputs'] and c['execution_count'] is None
print('PASS: exact uploaded recipe, V22 only gains source PubChemLite block, correct overlap exclusion and array alignment, BIO/input guards, cells and embedded modules compile')
