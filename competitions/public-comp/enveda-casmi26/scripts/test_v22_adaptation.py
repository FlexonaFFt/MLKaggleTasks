'''Execute the native E2R2 file checks and experimental wrapper with synthetic identities.'''
import ast
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from build_v22_adaptation import FRAMEWORK,UPLOAD,SOURCE,build

original = json.loads((SOURCE if SOURCE.exists() else UPLOAD).read_bytes())
env={}
exec(FRAMEWORK,env)
tree=ast.parse(''.join(original['cells'][36]['source']))
functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
exec(compile(functions,'native-functions','exec'),env)
identity=lambda s:None if not isinstance(s,str) or s in ('bad','') else s.replace('-alias','')
key=env['e2r2_make_key'](identity)
native=env['e2r2_row']
wrapper=env['MetricDistinctRunnerUp'](native)
assert wrapper.rank(['A','B','C'],['A','X'],key)[0]==['A','X','B','C']
assert wrapper.rank(['A','B'],['A-alias','A','X'],key)[0]==['A','X','B']
assert wrapper.rank(['A','B'],['A','A-alias','A'],key)==native(['A','B'],['A','A-alias','A'],key)
assert wrapper.rank(['A','B'],['A','A','A','X'],key)[0]==['A','B']
assert wrapper.rank(['A','B'],['Y','X'],key)==native(['A','B'],['Y','X'],key)
assert wrapper.rank(['A','B'],['bad','X'],key)==native(['A','B'],['bad','X'],key)
assert wrapper.rank(['CCO'],['A','X'],key)[0]==['CCO']
assert wrapper.rank(['A','B'],['A','X;Y','bad'],key)[0]==['A','B']
assert wrapper.rank(['A','X-alias','X','B'],['A','X'],key)[0]==['A','X-alias','B']
env['V22Experiment'].check_bio(['/kaggle/input/datasets/sergiomalv/casmi-chebi-lmsd-structures/bio_meta.pkl'])
for paths in ([],['/kaggle/input/old-bio/bio_meta.pkl'],['/kaggle/input/casmi-chebi-lmsd-structures/bio_meta.pkl','/kaggle/input/old/bio_meta.pkl']):
    try:
        env['V22Experiment'].check_bio(paths)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing/ambiguous input should fail')

base=pd.DataFrame([('agree','A;B;C'),('differ','A;B;C'),('inactive','CCO')],columns=['molecule_id','smiles'])
cwd=os.getcwd()
with tempfile.TemporaryDirectory() as directory:
    try:
        os.chdir(directory)
        base.to_csv('sample_submission.csv',index=False)
        for mode in (False,True):
            base.to_csv('submission.csv',index=False)
            pack={i:'' for i in (3,5,6,8,9,11,13,15,17,18,20,22,24,26,28,29,30,31,33,34)}
            pack[36]=''.join(original['cells'][36]['source'])
            trial=env['V22Experiment'](pack,runner_up=mode)
            trial.scope.update(COMP=directory,te=pd.DataFrame(dict(molecule_id=['agree','differ'])),
                IS_RERUN=False,metric_identity=identity,
                ENG=dict(agree=dict(smiles=['A','X']),differ=dict(smiles=['Y','Z'])))
            with patch.object(env['glob'],'glob',return_value=['/kaggle/input/casmi-chebi-lmsd-structures/bio_meta.pkl']):
                report=trial.run()
            control=pd.read_csv('submission_v22_control.csv',dtype=str).set_index('molecule_id').smiles
            final=pd.read_csv('submission.csv',dtype=str).set_index('molecule_id').smiles
            assert control['agree']=='A;B;C' and control['differ']=='A;Y;B;C'
            assert final['agree']==('A;X;B;C' if mode else 'A;B;C')
            assert final['differ']==control['differ'] and final['inactive']=='CCO'
            assert report['changed_rows_vs_v22']==int(mode)
            if mode:
                assert Path('runner_up_work/e2r2_stats.json').exists()
    finally:
        os.chdir(cwd)

for nb in build():
    node=next(n for n in ast.parse(''.join(nb['cells'][4]['source'])).body if isinstance(n,ast.Assign))
    pack=ast.literal_eval(node.value)
    for i,source in pack.items():
        expected=''.join(original['cells'][i]['source'])
        if i==9:
            expected=expected.replace("assert platform.python_version_tuple()[:2] == ('3', '12')",
                "assert platform.python_version_tuple()[:2] in (('3', '12'), ('3', '13'))")
        assert source==expected,i
        compile(source,f'source-{i}','exec')
    assert nb['metadata']['kaggle']['dataSources']==original['metadata']['kaggle']['dataSources']
    for cell in nb['cells']:
        compile(''.join(cell['source']),'cell','exec')
        assert not cell['outputs'] and cell['execution_count'] is None
print('PASS: native E2R2 integration, runner-up physical top3/metric twins/invalids, protected top1, separate paired control, BIO ambiguity rejection and source arithmetic preservation')
