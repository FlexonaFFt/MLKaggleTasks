"""Offline sequential baseline, fixed tabular members and CPU finalization."""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

assert os.name!='nt' and Path('/kaggle/input').is_dir(),'Kaggle execution only'
here=Path(__file__).resolve().parent
config=json.loads((here/'runtime_config.json').read_text())
assert config['reference_scored_script_version_id']==359313348
assert config['candidate_admitted'], 'Candidate admission required'
assert config['protected_weight']==.10 and config['raw_extension'] == 'smol_center_add'
assert not config['second_column_balance'] and config['probability_transform']=='none'
matches=list(Path('/kaggle/input').rglob(config['tabpfn']['checkpoint_filename']))
assert len(matches)==1,'Attach the exact reviewed offline tabular asset package'
asset=matches[0].parent
manifest=json.loads((asset/'runtime_manifest.json').read_text())
assert manifest['status']=='CPU_SETUP_COMPLETE_NO_MODEL_EXECUTION'
assert manifest['global_environment_unchanged'] and not manifest['model_fit'] and not manifest['model_predict']
assert manifest['checkpoint_sha256']==config['tabpfn']['checkpoint_sha256']
assert matches[0].stat().st_size==config['tabpfn']['checkpoint_bytes']
with matches[0].open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==config['tabpfn']['checkpoint_sha256']
lock=asset/'runtime-addon-lock.txt'
wheels=asset
assert lock.is_file() and any(wheels.glob('*.whl')),'Reviewed offline addon lock/wheels are required'
for item in manifest['selected_addons'].values():
    with (wheels/item['wheel']).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']
for line in lock.read_text().splitlines():
    if '==' in line and not line.lstrip().startswith('#'):
        package=re.split(r'==',line.strip(),maxsplit=1)[0].lower().replace('_','-')
        assert package not in ('torch','transformers','peft','accelerate','safetensors')
        assert not package.startswith('nvidia-'),'Reuse the original CUDA/Torch runtime'
Path('/kaggle/temp').mkdir(parents=True,exist_ok=True)
temp=Path(tempfile.mkdtemp(prefix='ntl-tabular-',dir='/kaggle/temp'))
site=temp/'site'
base_env=os.environ.copy()
base_env['NTL_TABULAR_TEMP']=str(temp)
base_env['NTL_TABULAR_ASSET']=str(asset)
subprocess.run([sys.executable,'-m','pip','install','--no-index','--find-links',str(wheels),
    '--no-deps','--require-hashes','--target',str(site),'-r',str(lock)],
    check=True,timeout=config['package_install_timeout_seconds'])
tabular_env=base_env.copy()
tabular_env['PYTHONPATH']=str(site)
tabular_env['HF_HUB_OFFLINE']='1'
subprocess.run([sys.executable,'-c',
    "import json, importlib.metadata as m; from pathlib import Path; import numpy, pandas, scipy, sklearn, torch; from tabpfn import TabPFNClassifier; from tabpfn.constants import ModelVersion; actual=dict(numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__, tabpfn=m.version('tabpfn')); expected="+repr(config['verified_runtime_imports'])+"; assert actual==expected, (actual,expected); assert ModelVersion.V3_5; Path('/kaggle/working/tabular_environment_audit.json').write_text(json.dumps(dict(status='VERIFIED_OFFLINE_RUNTIME_IMPORTS',imports=actual,diagnostic_script_version_id=359162368),indent=2)); print('Offline TabPFN imports passed', flush=True)"],
    env=tabular_env,check=True,timeout=90)
subprocess.run([sys.executable,str(here/'run_baseline.py')],cwd='/kaggle/working',env=base_env,
    check=True,timeout=config['baseline_timeout_seconds'])
baseline=Path('/kaggle/working/submission.parquet')
assert baseline.is_file()
control=Path('/kaggle/working/B556_gated_control.parquet')
assert not control.exists()
baseline.rename(control)
subprocess.run([sys.executable,str(here/'tabular_worker.py'),'prepare'],env=tabular_env,
    check=True,timeout=config['prepare_timeout_seconds'])
base_env['NTL_TABULAR_SITE']=str(site)
subprocess.run([sys.executable,'-u',str(here/'run_parallel.py')],env=base_env,check=True,
    timeout=max(config['model_timeout_seconds'],3600)+120)
subprocess.run([sys.executable,str(here/'finish_center_protected.py')],env=base_env,check=True,timeout=120)
assert Path('/kaggle/working/submission.parquet').is_file()
assert json.loads(Path('/kaggle/working/protected_residual_audit.json').read_text())['status']=='SMOL_CENTER_ADD_PROTECTED_PFN10_PREDICTIONS_COMPLETE'