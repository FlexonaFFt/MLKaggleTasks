'''Synthetic parity against the uploaded rank2 cell; verify exact source change boundaries.'''
import ast
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from build_pubchem_promotion_adaptation import FRAMEWORK,SOURCE,UPLOAD,build

original = json.loads((SOURCE if SOURCE.exists() else UPLOAD).read_bytes())
env = {}
exec(FRAMEWORK,env)
key = lambda s: None if s in ('bad','CCO','') else ('A' if s=='A-alias' else s)
hedge = env['EngineTwoHedge'](key)
assert hedge.rank(['A','B','C'],['B']) == ['A','B','C']
assert hedge.rank(['A','C','B'],['B']) == ['A','B','C']
assert hedge.rank(['A','C'],['A-alias']) == ['A','C']
assert hedge.rank(['A','C'],['bad']) == ['A','C']
assert len(hedge.rank([str(i) for i in range(25)],['extra'])) == 25
rows = [('absent','A;C'),('invalid','A;C'),('same','A;C'),('move','A;C;B'),('new','A;C'),('placeholder','CCO')]
eng = dict(invalid={'smiles':['bad']},same={'smiles':['A-alias']},move={'smiles':['B']},new={'smiles':['B']})
oracle_scope = dict(pd=pd,os=os,json=json,ENG=eng,chem=SimpleNamespace(score_key=key),
    _Chp=SimpleNamespace(MolFromSmiles=lambda s:None if s=='bad' else object()))
tree = ast.parse(''.join(original['cells'][47]['source']))
tree.body = [n for n in tree.body if not isinstance(n,ast.ImportFrom)]
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as directory:
    try:
        os.chdir(directory)
        pd.DataFrame(rows,columns=['molecule_id','smiles']).to_csv('submission.csv',index=False)
        pd.DataFrame(rows,columns=['molecule_id','smiles']).to_csv('sample_submission.csv',index=False)
        oracle_scope['COMP'] = directory
        exec(compile(tree,'native-rank2','exec'),oracle_scope)
        actual = pd.read_csv('submission.csv',dtype=str)
        expected = [';'.join(hedge.rank(s.split(';'),eng.get(mid,{}).get('smiles',[]))) for mid,s in rows]
        assert actual.smiles.tolist() == expected
    finally:
        os.chdir(cwd)

for threshold,nb in zip((100.,150.),build()):
    pack_node = next(n for n in ast.parse(''.join(nb['cells'][4]['source'])).body if isinstance(n,ast.Assign))
    pack = ast.literal_eval(pack_node.value)
    for i,source in pack.items():
        expected = ''.join(original['cells'][i]['source'])
        if i==10:
            expected = expected.replace("assert platform.python_version_tuple()[:2] == ('3', '12')",
                "assert platform.python_version_tuple()[:2] in (('3', '12'), ('3', '13'))")
        if i==32:
            expected = expected.replace('(rel > 100.0)',f'(rel > {threshold})').replace(
                'rel promotion threshold 100.0',f'rel promotion threshold {threshold}')
        assert source == expected, i
        compile(source,f'source-{i}','exec')
    assert nb['metadata']['kaggle']['dataSources'] == original['metadata']['kaggle']['dataSources']
    assert nb['metadata']['kaggle']['isGpuEnabled'] is True
    for cell in nb['cells']:
        compile(''.join(cell['source']),'cell','exec')
        assert not cell['outputs'] and cell['execution_count'] is None
print('PASS: raw-source/OOP rank2 parity, invalid/same/absent/dedup/tail cases, preserved input versions, exact threshold/runtime-only source edits')
