import numpy as np
K = 18
BALANCE_ALPHA = .5

def source_balance(LOGIT):
    P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    if BALANCE_ALPHA > 0 and len(P) >= 2:
        target = (1 - BALANCE_ALPHA) * P.sum(0) + BALANCE_ALPHA * np.full(K, len(P) / K)
        for _ in range(50):
            P *= (target / (P.sum(0) + 1e-12))[None, :]
            P /= P.sum(1, keepdims=True) + 1e-12
        print('balancing changed', int((P.argmax(1) != LOGIT.argmax(1)).sum()), 'of', len(P), flush=True)
    return P
