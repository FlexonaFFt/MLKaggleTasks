"""Fixed train-only feature preparation or one isolated tabular inference member."""
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path

assert os.name!='nt' and Path('/kaggle/input').is_dir()
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np

here=Path(__file__).resolve().parent
config=json.loads((here/'runtime_config.json').read_text())
cache=Path(os.environ['NTL_TABULAR_TEMP']).resolve()
assert cache.is_relative_to(Path('/kaggle/temp').resolve())
mode=sys.argv[1]
started=time.monotonic()

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def style24(t):
    # Original reference first 20 deterministic scalars, plus four fixed ratios.
    words = re.findall(r"[A-Za-z\u2019']+", t)
    nw, nc = max(len(words), 1), max(len(t), 1)
    sents = [x for x in re.split(r'(?<=[.!?])\s+', t.strip()) if x]
    sl = np.array([len(re.findall(r"[A-Za-z\u2019']+", x)) for x in sents]) if sents else np.array([0])
    wl = np.array([len(x) for x in words]) if words else np.array([0])
    lw = [x.lower() for x in words]
    return [np.log1p(nw), len(sents), sl.mean(), sl.std(), sl.max(), sl.min(), wl.mean(), wl.std(),
        (wl >= 9).mean(), len(set(lw))/nw, sum(c.isupper() for c in t)/nc, sum(c.isdigit() for c in t)/nc,
        sum(ord(c)>127 for c in t)/nc, t.count(', ')/nw, t.count(' and ')/nw, t.count(' but ')/nw,
        sum(x[:1].isupper() and x.split()[0].lower() in ('this','these','it','they') for x in sents)/max(len(sents),1),
        sum(x.endswith('ly') for x in words)/nw, sum(x.endswith('ing') for x in words)/nw,
        sum(x.endswith('tion') for x in words)/nw,
        float(any(c in t for c in ('\u2019','\u201c','\u201d'))), t.count('\u2014')/nw,
        t.count('\n')/nc, sum(bool(x.strip()) for x in t.splitlines())/nw]

with np.load(cache/'alignment.npz',allow_pickle=False) as saved:
    alignment={key:saved[key].copy() for key in saved.files}
if mode=='prepare':
    import pandas as pd
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    data=next(path for path in [Path('/kaggle/input/competitions/name-that-llm'),Path('/kaggle/input/name-that-llm')] if (path/'train.csv').is_file())
    train,test=pd.read_csv(data/'train.csv'),pd.read_csv(data/'test.csv')
    for frame,key in [(train,'train_ids'),(test,'test_ids')]:
        np.testing.assert_array_equal(frame.id.to_numpy(dtype=str),alignment[key])
    np.testing.assert_array_equal(train.model.to_numpy(dtype=str),alignment['labels'][alignment['y']])
    train_columns,test_columns,audit=[],[],[]
    for key,width in [('q3_8b_cat7+22',8192),('style4b',5120)]:
        train_view=np.load(cache/(key+'_train.npy'),mmap_mode='r',allow_pickle=False)
        test_view=np.load(cache/(key+'_test.npy'),mmap_mode='r',allow_pickle=False)
        assert train_view.shape==(len(train),width) and test_view.shape==(len(test),width)
        pca=PCA(n_components=64,svd_solver='randomized',iterated_power=3,random_state=42)
        pca.fit(train_view)
        train_columns.append(pca.transform(train_view).astype(np.float32))
        test_columns.append(pca.transform(test_view).astype(np.float32))
        audit.append(dict(view=key,fit_rows=len(train),components=64,fit_scope='official_train_only',
            explained_variance=float(pca.explained_variance_ratio_.sum())))
    for frame,columns in [(train,train_columns),(test,test_columns)]:
        texts=frame.text.fillna('').astype(str)
        columns.append(np.asarray([style24(text) for text in texts],dtype=np.float32))
    assert sha(here/'masked_features.py')==config['masked_feature_module_sha256']
    base_train,base_test=np.concatenate(train_columns,axis=1),np.concatenate(test_columns,axis=1)
    from masked_features import masked_svd_blocks
    text_train,text_test,text_audit=masked_svd_blocks(
        train.text.fillna('').astype(str).tolist(),test.text.fillna('').astype(str).tolist())
    train_columns.append(text_train)
    test_columns.append(text_test)
    x_train,x_test=np.concatenate(train_columns,axis=1),np.concatenate(test_columns,axis=1)
    assert x_train.shape==(len(train),184) and x_test.shape==(len(test),184)
    np.testing.assert_array_equal(x_train[:,:152],base_train)
    np.testing.assert_array_equal(x_test[:,:152],base_test)
    assert np.isfinite(x_train).all() and np.isfinite(x_test).all()
    np.savez(cache/'tables.npz',x_train=x_train,x_test=x_test,**alignment)
    Path('/kaggle/working/tabular_prepare_audit.json').write_text(json.dumps(dict(
        status='FULL_TRAIN_ONLY_PCA_STYLE24_AND_MASKED_SVD_COMPLETE',pca=audit,text_svd=text_audit,features=184,
        masked_feature_module_sha256=config['masked_feature_module_sha256'],
        query_rows=len(test),elapsed_seconds=time.monotonic()-started,
        packages={name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','scipy')},
        training_context_exported=False),indent=2))
    sys.exit(0)

assert mode=='tabpfn'
import torch
assert torch.cuda.is_available()
assert torch.cuda.mem_get_info()[0]>8*1024**3
with np.load(cache/'tables.npz',allow_pickle=False) as saved:
    x_train,x_test,y=(saved[key].copy() for key in ('x_train','x_test','y'))
asset=Path(os.environ['NTL_TABULAR_ASSET'])
settings=config[mode]
checkpoint=asset/settings['checkpoint_filename']
assert checkpoint.is_file() and sha(checkpoint)==settings['checkpoint_sha256']
assert importlib.metadata.version(mode)==settings['package_version']
assert settings['n_estimators']==128,'Preserve the selected128-estimator configuration'
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
params=dict(n_estimators=128,model_path=str(checkpoint),device='cuda:0',fit_mode='low_memory',
    memory_saving_mode='auto',balance_probabilities=True,tuning_config=None,random_state=42,n_preprocessing_jobs=1)
model=TabPFNClassifier.create_default_for_version(ModelVersion.V3_5,**params)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize();t=time.monotonic()
model.fit(x_train,y)
torch.cuda.synchronize();fit_seconds=time.monotonic()-t
t=time.monotonic();raw=model.predict_proba(x_test)
torch.cuda.synchronize();predict_seconds=time.monotonic()-t
classes=np.asarray(model.classes_,dtype=int)
np.testing.assert_array_equal(np.sort(classes),np.arange(18))
probabilities=np.zeros((len(x_test),18),dtype=np.float64)
probabilities[:,classes]=raw
assert np.isfinite(probabilities).all() and (probabilities>=0).all()
np.testing.assert_allclose(probabilities.sum(1),1,rtol=0,atol=1e-5)
np.savez(cache/(mode+'_proba.npz'),proba=probabilities,ids=alignment['test_ids'],labels=alignment['labels'])
Path('/kaggle/working/'+mode+'_inference_audit.json').write_text(json.dumps(dict(
    status='COMPLETE',model=mode,checkpoint_sha256=settings['checkpoint_sha256'],parameters=params,
    train_rows=len(x_train),query_rows=len(x_test),features=184,fit_seconds=fit_seconds,
    predict_seconds=predict_seconds,peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
    peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),probability_transform='none',
    row_mass_maxabs=float(np.max(np.abs(probabilities.sum(1)-1))),
    packages={name:importlib.metadata.version(name) for name in (mode,'torch','numpy','scikit-learn')},
    oof=False,model_context_exported=False),indent=2))
