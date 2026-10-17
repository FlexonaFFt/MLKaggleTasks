'''Single-factor gate and native E006/metric-exact hedge integration checks, no GPU.'''
import ast
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from build_v22_adaptation import FRAMEWORK,build as baseline_build
from build_v22_ablation_pair import FOLLOWUP,build
from build_flow_experiments import ROOT

env={}
exec(FRAMEWORK+FOLLOWUP,env)
base=baseline_build()[0]
node=next(n for n in ast.parse(''.join(base['cells'][4]['source'])).body if isinstance(n,ast.Assign))
sources=ast.literal_eval(node.value)
# Both the popularity OR branch and the separate slot threshold remain unchanged.
tree=ast.parse(sources[24])
condition=next(n.test for n in ast.walk(tree) if isinstance(n,ast.If) and 'S_TAU' in ast.unparse(n.test))
revised=ast.parse(ast.unparse(condition).replace('rel > 150.0','rel > 200.0'),mode='eval')
for popularity in (False,True):
    for rel in (149.,150.,151.,199.,200.,201.,1e9):
        context=dict(S=7. if popularity else None,S_TAU=6.,pop_top=6.,POP_TAU=5.,rel=rel)
        assert eval(compile(revised,'gate','eval'),context)==(popularity or rel>200.)

spec=importlib.util.spec_from_file_location('gl',ROOT/'archive/simulator-interface-2026-10-09/gl_fuse.py')
gl=importlib.util.module_from_spec(spec);spec.loader.exec_module(gl)
spec=importlib.util.spec_from_file_location('ice',ROOT/'archive/simulator-interface-2026-10-09/ice_fuse.py')
ice=importlib.util.module_from_spec(spec);spec.loader.exec_module(ice)
fusion_tree=ast.parse(sources[30])
fusion_fn=next(n for n in fusion_tree.body if isinstance(n,ast.FunctionDef) and n.name=='metric_fusion')
helpers={};exec(compile(ast.Module(body=[fusion_fn],type_ignores=[]),'fusion','exec'),helpers)
key=lambda s:s if isinstance(s,str) and s else None
cwd=os.getcwd()
with tempfile.TemporaryDirectory() as directory:
    try:
        os.chdir(directory)
        e004=pd.DataFrame([('m','A;B;C'),('inactive','CCO')],columns=['molecule_id','smiles'])
        e004.to_csv('sample_submission.csv',index=False)
        e004.to_csv('submission_reference.csv',index=False)
        e004.to_csv('submission.csv',index=False)
        e004_bytes=Path('submission.csv').read_bytes()
        Path('e004_verification.json').write_text(json.dumps(dict(experiment='E004',processed_ids=['m'],
            hashes={'submission.csv':hashlib.sha256(e004_bytes).hexdigest()})))
        scope=dict(pd=pd,COMP=directory,te=pd.DataFrame(dict(molecule_id=['m'])),ENG={'m':dict(smiles=['A','B','C'])},
            PREFUSION_LISTS={'m':['A','B','C']},metric_identity=key,metric_fusion=helpers['metric_fusion'],
            chem=SimpleNamespace(score_key=key),ALPHA=.6,KRR=3.,_form2=lambda s:'F',
            ICE_SCORES={},gl_of=lambda mid:{},GL_LAM=1.,ICE_LAM=1.,gl_fuse=gl,ice_fuse=ice,
            FRAG_GAP={'m':dict(A=0.,B=0.,C=10.)},IS_RERUN=False,
            safe_extend=lambda first,tail:list(dict.fromkeys(first+tail))[:25])
        env['V22Stage']('Native control',[34,36]).run(scope,sources)
        control_bytes=Path('submission.csv').read_bytes()
        control=pd.read_csv('submission.csv',dtype=str)
        assert control.iloc[0].smiles!='A;B;C', 'Synthetic fragment term should be active'
        trial=env['V22Ablation'](SimpleNamespace(scope=scope),sources,'no_fragment_gap')
        report=trial.run()
        assert Path('submission_v22_0451_control.csv').read_bytes()==control_bytes
        assert pd.read_csv('submission.csv',dtype=str).equals(e004)
        assert report['changed_rows']==1 and scope['FRAG_GAP']=={}
        # A broken E004 hash must fail before reconstruction, not silently rebuild from final rows.
        Path('e004_verification.json').write_text(json.dumps(dict(hashes={'submission.csv':'bad'})))
        try:
            trial.run()
        except AssertionError:
            pass
        else:
            raise AssertionError('Broken E004 control hash accepted')
        # Verify all rel200 descendants execute in order; no simulator/engine stage is repeated.
        marker_pack={i:"trace.append(%d)"%i for i in (26,29,30,31,33,34,36)}
        marker_pack[24]="trace.append(24)\nassert (rel > 150.0) is True\n"
        scope.update(trace=[],rel=201.)
        e004.to_csv('submission.csv',index=False)
        report=env['V22Ablation'](SimpleNamespace(scope=scope),marker_pack,'promotion_rel200').run()
        assert scope['trace']==[24,26,29,30,31,33,34,36]
        assert report['relative_threshold']==200. and report['changed_rows']==0
    finally:
        os.chdir(cwd)

for nb in build():
    assert nb['cells'][1:len(base['cells'])]==base['cells'][1:]
    assert nb['metadata']['kaggle']==base['metadata']['kaggle']
    for cell in nb['cells']:
        compile(''.join(cell['source']),'cell','exec')
        assert not cell['outputs'] and cell['execution_count'] is None
print('PASS: strict rel200/popularity boundaries, real E006 fragment ablation + metric-exact hedge, protected placeholders, control hashes, descendant order and unchanged V22 inputs/credits')
