"""Capture existing frozen features and balanced baseline into ephemeral storage."""
import hashlib
import json
import os
from functools import wraps
from pathlib import Path
import numpy as np
from claude_gate import save_gate_state

VIEWS = {'q3_8b_cat7+22':8192, 'style4b':5120}

def cache_directory():
    directory = Path(os.environ['NTL_TABULAR_TEMP']).resolve()
    assert directory.is_relative_to(Path('/kaggle/temp').resolve())
    directory.mkdir(parents=True,exist_ok=True)
    return directory

def wrap_fit_head(original, train, test, labels):
    cache = cache_directory()
    np.savez_compressed(cache/'alignment.npz',
        train_ids=train.id.to_numpy(dtype=str), test_ids=test.id.to_numpy(dtype=str),
        labels=np.asarray(labels,dtype=str),
        y=train.model.map({label:i for i,label in enumerate(labels)}).to_numpy(dtype=np.int64))
    captured = set()
    @wraps(original)
    def fitting(key, Etr, Ete, C):
        if key in VIEWS:
            assert key not in captured
            assert Etr.dtype==Ete.dtype==np.float32
            assert Etr.shape==(len(train),VIEWS[key]) and Ete.shape==(len(test),VIEWS[key])
            assert np.isfinite(Etr).all() and np.isfinite(Ete).all()
            for suffix,value in [('train',Etr),('test',Ete)]:
                np.save(cache/(key+'_'+suffix+'.npy'),value,allow_pickle=False)
            captured.add(key)
            (cache/'captured_views.json').write_text(json.dumps(sorted(captured)))
        return original(key,Etr,Ete,C)
    return fitting

def capture_balanced(P,test,labels,DC_TE,LC_TE,CLAUDE_IDX,gate_lam,lora_lam):
    cache=cache_directory()
    assert set(json.loads((cache/'captured_views.json').read_text()))==set(VIEWS)
    assert P.dtype==np.float64 and P.shape==(len(test),len(labels))
    assert np.isfinite(P).all() and (P>=0).all()
    np.savez_compressed(cache/'balanced_baseline.npz',probabilities=P,
        ids=test.id.to_numpy(dtype=str),labels=np.asarray(labels,dtype=str))
    save_gate_state(cache,P,test.id.to_numpy(dtype=str),labels,
        DC_TE,LC_TE,CLAUDE_IDX,gate_lam,lora_lam)
    records={}
    for path in sorted(cache.glob('*.npy')):
        with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        records[path.name]=dict(bytes=path.stat().st_size,sha256=digest)
    # Only sizes and hashes are exportable; arrays and targets stay under /kaggle/temp.
    Path('/kaggle/working/tabular_capture_audit.json').write_text(json.dumps(dict(
        status='TWO_EXISTING_VIEWS_CAPTURED_WITHOUT_EXTRA_BACKBONE_PASS',
        train_rows=len(np.load(cache/'alignment.npz',allow_pickle=False)['train_ids']),
        query_rows=len(test),views=records,baseline_classes=list(labels),
        source='exact ntl-v6 v2 raw fit_head inputs before preprocessing',
        gate_state_captured_before_original_argmax=True,
        dc_dtype=None if DC_TE is None else str(DC_TE.dtype),
        lc_dtype=None if LC_TE is None else str(LC_TE.dtype),
        claude_indices_dtype=str(CLAUDE_IDX.dtype),
        ephemeral_training_arrays=True),indent=2))
