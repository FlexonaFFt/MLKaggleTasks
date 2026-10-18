"""Exact ntl-v6 hard Claude decision, with lossless temporary state transport."""
from pathlib import Path
import numpy as np


def validate_gate(P, labels, DC_TE, LC_TE, CLAUDE_IDX, gate_lam, lora_lam):
    labels = np.asarray(labels)
    assert labels.ndim == 1 and len(labels) == len(np.unique(labels)) == 18
    assert P.dtype == np.float64 and P.shape[1:] == (len(labels),)
    assert np.isfinite(P).all() and (P >= 0).all()
    expected = np.array([i for i, label in enumerate(labels) if label.startswith('claude')])
    assert len(expected) == 5
    assert np.issubdtype(CLAUDE_IDX.dtype, np.integer)
    np.testing.assert_array_equal(CLAUDE_IDX, expected)
    assert (gate_lam, lora_lam) == (1.5, 1.0)
    for value in (DC_TE, LC_TE):
        if value is not None:
            assert np.issubdtype(value.dtype, np.floating)
            assert value.shape == (len(P), len(CLAUDE_IDX)) and np.isfinite(value).all()


def gated_predict(P, labels, DC_TE, LC_TE, CLAUDE_IDX, gate_lam=1.5, lora_lam=1.0):
    validate_gate(P, labels, DC_TE, LC_TE, CLAUDE_IDX, gate_lam, lora_lam)
    # Keep the reference operations, dtypes, clipping, tie order and optional gate exactly.
    pred = P.argmax(1)
    if DC_TE is not None:
        _isc = np.isin(pred, CLAUDE_IDX)
        _s = np.log(np.clip(P[:, CLAUDE_IDX], 1e-12, 1)) + gate_lam * DC_TE
        if LC_TE is not None:
            _s = _s + lora_lam * LC_TE
        _new = CLAUDE_IDX[_s.argmax(1)]
        pred[_isc] = _new[_isc]
    return pred


def save_gate_state(directory, P, ids, labels, DC_TE, LC_TE, CLAUDE_IDX, gate_lam, lora_lam):
    validate_gate(P, labels, DC_TE, LC_TE, CLAUDE_IDX, gate_lam, lora_lam)
    # Explicit flags avoid object arrays and preserve the actual present-array dtypes.
    np.savez_compressed(Path(directory)/'claude_gate_state.npz',
        ids=np.asarray(ids, dtype=str), labels=np.asarray(labels, dtype=str),
        dc_present=np.asarray(DC_TE is not None), lc_present=np.asarray(LC_TE is not None),
        DC_TE=DC_TE if DC_TE is not None else np.empty((0, 5), dtype=np.float64),
        LC_TE=LC_TE if LC_TE is not None else np.empty((0, 5), dtype=np.float64),
        CLAUDE_IDX=CLAUDE_IDX, gate_lam=np.asarray(gate_lam), lora_lam=np.asarray(lora_lam))


def load_gate_state(directory, P, ids, labels):
    with np.load(Path(directory)/'claude_gate_state.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['ids'], ids)
        np.testing.assert_array_equal(saved['labels'], labels)
        state = dict(DC_TE=saved['DC_TE'].copy() if bool(saved['dc_present']) else None,
            LC_TE=saved['LC_TE'].copy() if bool(saved['lc_present']) else None,
            CLAUDE_IDX=saved['CLAUDE_IDX'].copy(),
            gate_lam=saved['gate_lam'].item(), lora_lam=saved['lora_lam'].item())
    validate_gate(P, labels, **state)
    return state
