import numpy as np

def source_logsoftmax(value):
    value = value - value.max(1, keepdims=True)
    return value - np.log(np.exp(value).sum(1, keepdims=True))


def add_logprob(p):
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(-1), 1, rtol=0, atol=2e-5)
    return np.log(np.clip(p, 1e-9, 1))


def lora_readouts(Z, ro, pools, recovered, adaptive):
    """Production FP32 affine and pool-local query centering, never global centering."""
    assert Z.dtype == np.float32 and Z.shape == (len(Z), 12288)
    assert ro['plain_coef'].dtype == ro['center_coef'].dtype == np.float32
    lg = Z @ ro['plain_coef'].T + ro['plain_intercept']
    lg -= lg.max(1, keepdims=True)
    p = np.exp(lg)
    plain = add_logprob(p / p.sum(1, keepdims=True))
    centered = np.zeros((len(pools), len(Z), 18), dtype=np.float32)
    for context in range(len(pools)):
        for pool in np.unique(pools[context]):
            local = np.flatnonzero((pools[context] == pool) & adaptive[context])
            if not len(local):
                continue
            X = np.asarray(Z[local], dtype=np.float32).copy()
            groups = recovered[context, local]
            for group in np.unique(groups):
                i = groups == group
                X[i] -= X[i].mean(0)
            lc = X @ ro['center_coef'].T + ro['center_intercept']
            centered[context, local] = source_logsoftmax(lc)
    LC = None
    if 'claude_coef' in ro:
        LC = source_logsoftmax(Z @ ro['claude_coef'].T + ro['claude_intercept'])
    return plain, centered, LC


