'''Exercise both final notebook cells with real CSV writes, without model inference.'''
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from build_pubchemlite_hedge_pair import build, source_build

base=source_build()[0]
cwd=Path.cwd()
for mode,nb in zip(('metric_exact_rank2','no_final_rank2'),build()):
    assert nb['cells'][2:6]==base['cells'][2:6]
    for cell in nb['cells']:
        compile(''.join(cell['source']),'notebook-cell','exec')
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            before=pd.DataFrame([('1','A;C;B'),('2','A;C'),('3','CCO')],columns=['molecule_id','smiles'])
            native=before.copy()
            native['smiles']=['A;B;C','A;A-alias;C','CCO']
            before.to_csv('submission_e006.csv',index=False)
            native.to_csv('submission.csv',index=False)
            before.to_csv('sample_submission.csv',index=False)
            exact=Path('submission_e006.csv').read_bytes()
            identity=lambda s: None if s=='bad' else ('A' if s=='A-alias' else s)
            scope=dict(COMP=directory,te=before.iloc[:2],metric_identity=identity,
                ENG={'1':{'smiles':['B']},'2':{'smiles':['A-alias']}})
            env=dict(pd=pd,Path=Path,json=json,hashlib=hashlib,experiment=SimpleNamespace(scope=scope))
            exec(''.join(nb['cells'][-1]['source']),env)
            final=pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
            expected=['A;B;C','A;C','CCO'] if mode=='metric_exact_rank2' else before.smiles.tolist()
            assert final.smiles.tolist()==expected
            if mode=='no_final_rank2':
                assert Path('submission.csv').read_bytes()==exact
            assert Path('pubchemlite_hedge_audit.json').is_file()
            saved=Path('submission.csv').read_bytes()
            checker=env['CheckedSubmission'](before,native,{'1','2'},identity)
            with patch('os.replace',side_effect=OSError('simulated write failure')):
                try:
                    checker.save(final,'submission.csv')
                except OSError:
                    pass
                else:
                    raise AssertionError('Failure was swallowed')
            assert Path('submission.csv').read_bytes()==saved
            assert not Path('submission_checked.tmp.csv').exists()
            invalid=final.copy();invalid.loc[0,'smiles']='Z;B'
            try:
                checker.save(invalid,'submission.csv')
            except AssertionError:
                pass
            else:
                raise AssertionError('Changed top1 accepted')
            assert Path('submission.csv').read_bytes()==saved
        finally:
            os.chdir(cwd)
print('PASS: both complete final cells, alias handling, unchanged top1/placeholders, exact E006 bytes, CSV readback and failure preservation; upstream inference not run')
