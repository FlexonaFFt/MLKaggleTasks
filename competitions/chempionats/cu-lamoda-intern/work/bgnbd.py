"""BG/NBD (Fader-Hardie-Lee 2005) для Lamoda repeat-purchase.

Сводка на отсечении (scoring_date - 14d) для каждого клиента:
  x  = число повторных покупок (n_orders - 1)
  t_x = дней от первой покупки до последней
  T   = возраст клиента в днях (первая покупка -> отсечение)
Прогноз: E[X(T+t0, T+t0+win)] с t0=14 (лаг), win=3 месяца.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

DATA = Path("/Users/flexonafft/MLKaggleTasks/competitions/chempionats/cu-lamoda-intern/data")
GAP = 14
train = pd.read_csv(DATA / "train.csv", parse_dates=["scoring_date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["scoring_date"])
orders = pd.read_parquet(DATA / "orders.parquet")
orders["order_date"] = pd.to_datetime(orders["order_date"])
od = orders.groupby(["customer_id", "order_date"]).size().reset_index()[["customer_id", "order_date"]]


def summary(cust):
    m = cust.merge(od, on="customer_id", how="left")
    cutoff = m.scoring_date - pd.Timedelta(days=GAP)
    m = m[m.order_date <= cutoff]
    g = m.groupby("customer_id").order_date.agg(["min", "max", "count"])
    s = cust.set_index("customer_id").copy()
    s["first"] = g["min"]
    s["last"] = g["max"]
    s["x"] = g["count"].fillna(0).astype(int) - 1  # повторные
    s["t_x"] = (g["max"] - g["min"]).dt.days
    s["age"] = (cust.set_index("customer_id").scoring_date - pd.Timedelta(days=GAP) - g["min"]).dt.days
    s["win"] = (cust.set_index("customer_id").scoring_date + pd.DateOffset(months=3)
                - cust.set_index("customer_id").scoring_date).dt.days
    return s


def ap20(target, score, ids):
    df = pd.DataFrame({"id": ids.astype(str), "t": target, "s": score}).sort_values(
        ["s", "id"], ascending=[False, True], kind="mergesort")
    k = int(np.ceil(len(df) * 0.2))
    top = df.t.to_numpy()[:k]
    return float((np.cumsum(top) / np.arange(1, k + 1) * top).sum() / max(min(k, int(df.t.sum())), 1))


# ---------- BG/NBD ----------
def log_likelihood(params, x, t_x, T):
    r, alpha, a, b = params
    if min(params) <= 0 or a <= 1.001:
        return -np.inf
    x = x.astype(float)
    eps = 1e-10
    # ln likelihood по одному клиенту (paper eq.7), x>0 и x=0 в одной формуле:
    # L = A1*A2 * [ (alpha+T)^-(r+x) + I(x>0) * a/(b+x-1) * (alpha+t_x)^-(r+x) ]
    ll = (gammaln(r + x) - gammaln(r) + r * np.log(alpha)
          + gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
          -(r + x) * np.log(alpha + T))
    term2 = np.where(x > 0,
                     np.log(a) - np.log(b + np.maximum(x - 1, eps)) + (r + x) * (np.log(alpha + T) - np.log(alpha + t_x)),
                     -np.inf)
    ll = ll + np.logaddexp(0.0, term2)
    return ll.sum()


def fit_bgnbd(x, t_x, T):
    from scipy.optimize import minimize as _min
    best = None
    for start in [(1.0, 10.0, 1.1, 1.0), (0.5, 30.0, 1.5, 2.0), (2.0, 5.0, 1.2, 0.5), (1.0, 60.0, 2.0, 2.0)]:
        res = _min(lambda p: -log_likelihood(p, x, t_x, T), np.array(start, dtype=float),
                   method="L-BFGS-B", bounds=[(1e-3, 50), (1e-2, 500), (1.001, 50), (1e-3, 200)])
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def p_alive(params, x, t_x, T):
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(x == 0,
                       1 + (a / b) * ((alpha + T) / alpha) ** r,
                       1 + (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x))
    return 1.0 / den


def expected_in(params, t, x, t_x, T):
    """E[X(T, T+t)] по paper eq.10 (lifetimes)."""
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
    first = (a + b + x - 1) / (a - 1)
    second = (1 - (alpha + T) / (alpha + T + t)) ** (r + x) * hyp
    den = 1 + (x > 0) * (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)
    return (r + x) / (alpha + T) * first * second / den


tr_sum = summary(train)
te_sum = summary(test)
print("customers with x=0 in train:", (tr_sum.x == 0).mean().round(3))

mask = tr_sum["age"].notna()
mvals = tr_sum[mask]
params = fit_bgnbd(mvals.x.to_numpy(), mvals.t_x.fillna(0).to_numpy(), mvals["age"].to_numpy())
print("BG/NBD params (r, alpha, a, b):", params.round(4))


def bgnbd_scores(s):
    x = s.x.to_numpy(float)
    t_x = s.t_x.fillna(0).to_numpy(float)
    T = s["age"].to_numpy(float)
    win = s.win.to_numpy(float)
    pa = p_alive(params, x, t_x, T)
    e_win = expected_in(params, GAP + win, x, t_x, T) - expected_in(params, GAP, x, t_x, T)
    return pa, 1 - np.exp(-np.clip(e_win, 0, None)), e_win


for hold in ["2022-02-01", "2022-03-01"]:
    s = tr_sum[tr_sum.scoring_date == hold]
    pa, pp, ee = bgnbd_scores(s)
    y = s.target.to_numpy()
    ids = s.index.to_numpy()
    print(f"\n== holdout {hold} (n={len(s)}, base={y.mean():.3f}) ==")
    print(f"  P(alive)      AP@20% = {ap20(y, pa, ids):.5f}")
    print(f"  P(>=1 покупки) AP@20% = {ap20(y, pp, ids):.5f}")
    print(f"  E[count]      AP@20% = {ap20(y, ee, ids):.5f}")
    print(f"  -E[count]     AP@20% = {ap20(y, -ee, ids):.5f}  (check direction)")
