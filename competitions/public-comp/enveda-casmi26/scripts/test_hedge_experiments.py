'''Policy boundaries and paired CSV integration without GPU or chemical model mocks.'''
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from build_pubchem_promotion_adaptation import FRAMEWORK,build as base_build
from build_hedge_experiments import HEDGES,build

env = {}
exec(FRAMEWORK+HEDGES,env)
# Deliberately different metric/source aliases prove the support gate uses official identity.
key = lambda s:None if s in ('bad','CCO','') else s.replace('-source-alias','')
identity = lambda s:None if s in ('bad','CCO','') else s.replace('-metric-alias','').replace('-source-alias','')
third = env['RankThreeHedge'](key)
second = env['SupportedRankTwoHedge'](key,identity)
assert third.rank(['A','C','D'],['B']) == ['A','C','B','D']
assert third.rank(['A','C','D','B'],['B']) == ['A','C','B','D']
assert third.rank(['A','B','C'],['B-source-alias']) == ['A','B','C']
assert third.rank(['A','C'],['A']) == ['A','C']
assert third.rank(['A'],[]) == ['A']
assert third.rank(['A','C'],['bad']) == ['A','C']
assert third.rank(['A'],['B']) == ['A','B']
assert third.rank([str(i) for i in range(25)],['B']) == ['0','1','B']+[str(i) for i in range(2,24)]
assert second.rank(['A','C'],['B'],['B-metric-alias']) == ['A','B','C']
assert second.rank(['A','C'],['B'],['D']) == ['A','C']
assert second.supported(['B'],[str(i) for i in range(9)]+['B-metric-alias'])
assert not second.supported(['B'],[str(i) for i in range(10)]+['B-metric-alias'])
assert not second.supported(['bad'],['bad'])
assert not second.supported([],['B'])

base = pd.DataFrame([('supported','A;C;D'),('unsupported','A;C;D'),('already2','A;B;C'),
    ('same','A;C;D'),('inactive','CCO')],columns=['molecule_id','smiles'])
eng = {mid:dict(smiles=[smile]) for mid,smile in [('supported','B'),('unsupported','B'),('already2','B'),('same','A')]}
source = env['EngineTwoHedge'](key)
control = base.copy()
control['smiles'] = [';'.join(source.rank(row.smiles.split(';'),eng.get(row.molecule_id,{}).get('smiles',[])))
    for row in base.itertuples(index=False)]
scope = dict(te=pd.DataFrame(dict(molecule_id=list(eng))),ENG=eng,_keyp=key,metric_identity=identity,
    PREFUSION_LISTS=dict(supported=['B-metric-alias'],unsupported=['D'],already2=['B'],same=['A']))
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as directory:
    try:
        os.chdir(directory)
        for mode in ('rank3','supported_rank2'):
            base.to_csv('submission_e006.csv',index=False)
            control.to_csv('submission.csv',index=False)
            raw = Path('submission.csv').read_bytes()
            report = env['HedgeExperiment'](SimpleNamespace(scope=scope),mode).run()
            result = pd.read_csv('submission.csv',dtype=str).set_index('molecule_id').smiles.to_dict()
            assert Path('submission_rel100_rank2_control.csv').read_bytes() == raw
            assert result['inactive'] == 'CCO' and result['already2'] == 'A;B;C'
            assert result['same'] == 'A;C;D'
            if mode=='rank3':
                assert result['supported'] == result['unsupported'] == 'A;C;B;D'
                assert report['changed_rows_vs_rank2'] == 2
            else:
                assert result['supported'] == 'A;B;C;D' and result['unsupported'] == 'A;C;D'
                assert report['changed_rows_vs_rank2'] == 1
            assert report['changed_top1'] == 0
            audit = pd.read_csv('hedge_comparison.csv')
            assert not audit.changed_top1.any() and len(audit)==len(base)
        # All support removed: conditional treatment reverts exactly to E006.
        scope['PREFUSION_LISTS'] = {}
        control.to_csv('submission.csv',index=False)
        env['HedgeExperiment'](SimpleNamespace(scope=scope),'supported_rank2').run()
        assert pd.read_csv('submission.csv',dtype=str).equals(base)
        # All source lists already agree: deterministic no-op reported to avoid duplicate submissions.
        scope['ENG'] = {mid:dict(smiles=['A']) for mid in eng}
        base.to_csv('submission.csv',index=False)
        report = env['HedgeExperiment'](SimpleNamespace(scope=scope),'rank3').run()
        assert report['changed_rows_vs_rank2'] == 0
    finally:
        os.chdir(cwd)

baseline = base_build()[0]
for nb in build():
    assert nb['cells'][1:len(baseline['cells'])] == baseline['cells'][1:]
    assert nb['metadata']['kaggle'] == baseline['metadata']['kaggle']
    assert nb['metadata']['casmi_experiment']['relative_threshold'] == 100.
    for cell in nb['cells']:
        compile(''.join(cell['source']),'notebook-cell','exec')
        assert cell['execution_count'] is None and not cell['outputs']
print('PASS: rank3 top2 shield, metric-identity top10 boundary, complete CSV treatments, source control bytes, no-op and unchanged rel100 recipe/inputs')
