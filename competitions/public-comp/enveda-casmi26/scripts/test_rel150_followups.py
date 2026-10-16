'''Native merge/rank2 fixtures and strict OOP gate boundaries without GPU inference.'''
import ast
import copy
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from build_rel150_followups import CAPTURE_FRAMEWORK,FOLLOWUPS,build
from build_pubchem_promotion_adaptation import build as base_build

env = {}
exec(CAPTURE_FRAMEWORK+FOLLOWUPS,env)
identity = lambda s:None if s in (None,'bad') else s.replace('-alias','')
policy = env['CandidateAlignedPromotion'](dict(m='X'),identity)
assert policy.allow('m',dict(pc=['X-alias']),True)
assert not policy.allow('m',dict(pc=['Y']),True)
assert not policy.allow('m',dict(pc=['X']),False)
assert not policy.allow('missing',dict(pc=['X']),True)
assert not env['CandidateAlignedPromotion'](dict(m='bad'),identity).allow('m',dict(pc=['bad']),True)
scope = dict(PC={'m':dict(pc=['X','Y'])})
env['CreditedStage']('Snapshot test',[29]).run(scope,{29:"PC['m']['pc'].reverse()"})
assert scope['PC_ORIGINAL_LEADERS']=={'m':'X'} and scope['PC']['m']['pc'][0]=='Y'

base = base_build()[1]
node = next(n for n in ast.parse(''.join(base['cells'][4]['source'])).body if isinstance(n,ast.Assign))
pack = ast.literal_eval(node.value)
old = '(S is not None and float(S) > S_TAU and pop_top >= POP_TAU)'
patched = pack[32].replace(old,'PROMOTION_ALIGNMENT.allow(mid,p,'+old+')')
# Exact source OR semantics: rel is strict >150, independent of identity alignment.
condition = next(n.test for n in ast.walk(ast.parse(patched)) if isinstance(n,ast.If)
    and 'PROMOTION_ALIGNMENT.allow' in ast.unparse(n.test))
for aligned in (False,True):
    for rel in (149.,150.,151.,1e9):
        for popularity in (False,True):
            gate = env['CandidateAlignedPromotion'](dict(m='X'),identity)
            p = dict(pc=['X' if aligned else 'Y'])
            context = dict(PROMOTION_ALIGNMENT=gate,mid='m',p=p,S=7. if popularity else None,
                pop_top=6.,S_TAU=6.,POP_TAU=5.,rel=rel)
            actual = eval(compile(ast.Expression(condition),'gate','eval'),context)
            assert actual == ((aligned and popularity) or rel>150.)

native_rank2 = ast.parse(pack[47])
native_rank2.body = [n for n in native_rank2.body if not isinstance(n,ast.ImportFrom)]
native_rank2 = ast.unparse(native_rank2)
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as directory:
    try:
        os.chdir(directory)
        mids = ['changed','margin','same','library']
        sample = pd.DataFrame(dict(molecule_id=mids+['inactive'],smiles=['CCO']*5))
        sample.to_csv('sample_submission.csv',index=False)
        key = identity
        fixture = dict(__name__='__main__',pd=pd,np=np,json=json,os=os,time=time,Path=Path,
            COMP=directory,te=pd.DataFrame(dict(molecule_id=mids)),
            BASE={m:(['A','B','C'],['A','B','C'],.95 if m=='library' else .1) for m in mids},
            PC={m:dict(pc=['X' if m=='same' else 'Y'],pc_keys=['X' if m=='same' else 'Y'],
                pc_fz=[0.],fz_top=151. if m=='margin' else 100.,best_pool_fz=0.,S=7.,top_pop=6.) for m in mids},
            PC_ORIGINAL_LEADERS={m:'X' for m in mids},
            LIB_TAU=.9,REL_TH=600.,SLOTS_AGG=[2,4,6,8,10],SLOTS_GENTLE=[4,8,12,16,20],
            CLAW_ICE_HEALTH='ok',CLAW_GL_HEALTH='ok',T0=time.time(),n_err=0,MAN=dict(families=[]),dev='cpu',
            ENG={},chem=SimpleNamespace(score_key=key),_Chp=SimpleNamespace(MolFromSmiles=lambda s:object()),
            metric_identity=identity)
        fixture['ENG'] = {m:dict(smiles=['B']) for m in mids}
        exec(pack[32],fixture)
        exec(native_rank2,fixture)
        control_bytes = Path('submission.csv').read_bytes()
        prehedge_bytes = Path('submission_e006.csv').read_bytes()
        # No final hedge restores the exact source E006 bytes, preserves top1 and placeholders.
        follow = env['Rel150Followup'](SimpleNamespace(scope=fixture),pack,'no_rank2')
        report = follow.run()
        assert Path('submission.csv').read_bytes()==prehedge_bytes
        assert Path('submission_rel150_rank2_control.csv').read_bytes()==control_bytes
        assert report['changed_rows']>0 and report['changed_top1']==0
        # Real native merge + native rank2; intervening numerical stages are NOT GPU-tested here.
        Path('submission.csv').write_bytes(control_bytes)
        test_pack = {i:'' for i in pack}
        test_pack[32],test_pack[47] = pack[32],native_rank2
        follow = env['Rel150Followup'](SimpleNamespace(scope=fixture),test_pack,'aligned_promotion')
        report = follow.run()
        result = pd.read_csv('submission.csv',dtype=str).set_index('molecule_id').smiles
        assert result['changed'].split(';')[0]=='A'
        assert result['margin'].split(';')[0]=='Y'
        assert result['same'].split(';')[0]=='X'
        assert result['library'].split(';')[0]=='A'
        assert result['inactive']=='CCO'
        assert report['changed_rows']==1 and report['changed_top1']==1
        assert report['popularity_clauses_blocked']==2
        assert Path('submission_rel150_rank2_control.csv').read_bytes()==control_bytes
    finally:
        os.chdir(cwd)

for nb in build():
    assert nb['metadata']['kaggle'] == base['metadata']['kaggle']
    assert nb['cells'][4] == base['cells'][4], 'Numerical source pack changed'
    assert nb['cells'][1:3] == base['cells'][1:3], 'Credits or inputs changed'
    for cell in nb['cells']:
        compile(''.join(cell['source']),'notebook-cell','exec')
        assert cell['execution_count'] is None and not cell['outputs']
print('PASS: pre-simulator immutable snapshot, metric aliases/missing identity, strict rel OR truth table, native merge/rank2 fixtures, byte-exact no-hedge restoration, preserved source/credits/inputs')
