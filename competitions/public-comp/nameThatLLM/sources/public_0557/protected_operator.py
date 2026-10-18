import numpy as np

def protected(q,p,baseline,claude,nonclaude,weight=.10):
    protected = np.isin(baseline, claude)
    np.testing.assert_array_equal(protected, np.isin(q.argmax(-1), claude))
    candidate = q.copy()
    reverted = np.zeros(q.shape[:2], dtype=bool)
    for context in range(q.shape[0]):
        rows = np.flatnonzero(~protected[context])
        original_non = q[context][np.ix_(rows, nonclaude)]
        non_mass = original_non.sum(1, keepdims=True)
        p_non = p[np.ix_(rows, nonclaude)]
        assert (non_mass > 0).all() and (p_non.sum(1) > 0).all()
        conditional = np.power(original_non / non_mass, 1 - weight) * np.power(p_non / p_non.sum(1, keepdims=True), weight)
        assert np.isfinite(conditional).all() and (conditional.sum(1) > 0).all()
        conditional /= conditional.sum(1, keepdims=True)
        candidate[context][np.ix_(rows, nonclaude)] = non_mass * conditional
        crossed = np.isin(candidate[context, rows].argmax(-1), claude)
        reverted[context, rows[crossed]] = True
        candidate[context, rows[crossed]] = q[context, rows[crossed]]
    # Retain original expert-selected Claude labels; only eligible non-Claude argmax labels may change.
    prediction = baseline.copy()
    prediction[~protected] = candidate.argmax(-1)[~protected]
    assert candidate.dtype == np.float64 and np.isfinite(candidate).all() and (candidate >= 0).all()
    assert candidate[:, :, claude].tobytes() == q[:, :, claude].tobytes(), "Claude columns changed at the bit level"
    assert candidate[protected].tobytes() == q[protected].tobytes(), "Protected whole row changed"
    assert candidate[reverted].tobytes() == q[reverted].tobytes(), "Family flip was not fully reverted"
    np.testing.assert_array_equal(prediction[protected], baseline[protected])
    np.testing.assert_array_equal(np.isin(prediction, claude), protected)
    np.testing.assert_allclose(candidate[:, :, nonclaude].sum(-1), q[:, :, nonclaude].sum(-1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(candidate.sum(-1), q.sum(-1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(candidate.sum(-1), 1, rtol=0, atol=2e-12)
    return candidate,prediction,reverted
