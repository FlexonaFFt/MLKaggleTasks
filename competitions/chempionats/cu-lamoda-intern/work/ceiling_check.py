"""Потолок сигнала: эмпирические условные вероятности таргета.
1) P(y | recency бакет, n_orders бакет) на train-доле -> apply на holdout (март).
2) P(y | recency, n_orders, median_interval) через сглаживание.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("/Users/flexonafft/MLKaggleTasks/competitions/chempionats/cu-lamoda-intern/data")
GAP = 14
train = pd.read_csv(DATA / "train.csv", parse_dates=["scoring_date"])
orders = pd.read_parquet(DATA / "orders.parquet")
orders["order_date"] = pd.to_datetime(orders["order_date"])
od = orders.groupby(["customer_id", "order_date"]).size().reset_index()[["customer_id", "order_date"]]


def base_feats(cust):
    m = cust.merge(od, on="customer_id", how="left")
    cutoff = m.scoring_date - pd.Timedelta(days=GAP)
    m = m[m.order_date <= cutoff]
    g = m.groupby("customer_id").order_date.agg(last="max", first="min", n="count")
    s = cust.set_index("customer_id").copy()
    s["rec"] = (s.scoring_date - pd.Timedelta(days=GAP) - g["last"]).dt.days
    dts = m.groupby("customer_id").order_date.apply(lambda x: x.sort_values().drop_duplicates().diff().dt.days.dropna())
    iv = dts.groupby("customer_id").median().rename("med_int")
    s = s.join(g[["n"]]).join(iv)
    s["rec"] = s["rec"].fillna(9999)
    s["n"] = s["n"].fillna(0)
    s["med_int"] = s["med_int"].fillna(9999)
    return s.reset_index()


def ap20(target, score, ids):
    df = pd.DataFrame({"id": ids.astype(str), "t": target, "s": score}).sort_values(
        ["s", "id"], ascending=[False, True], kind="mergesort")
    k = int(np.ceil(len(df) * 0.2))
    top = df.t.to_numpy()[:k]
    return float((np.cumsum(top) / np.arange(1, k + 1) * top).sum() / max(min(k, int(df.t.sum())), 1))


F = base_feats(train)
hold = F[F.scoring_date == "2022-03-01"].copy()
prior = F[F.scoring_date < "2022-03-01"].copy()

# бакеты
prior["r_b"] = pd.cut(prior.rec, [-1, 20, 30, 45, 60, 90, 150, 300, 9999])
prior["n_b"] = pd.cut(prior.n, [-1, 1, 2, 3, 5, 8, 15, 100])
hold["r_b"] = pd.cut(hold.rec, [-1, 20, 30, 45, 60, 90, 150, 300, 9999])
hold["n_b"] = pd.cut(hold.n, [-1, 1, 2, 3, 5, 8, 15, 100])

gmean = prior.target.mean()
table = prior.groupby(["r_b", "n_b"], observed=True).target.agg(["mean", "count"])
smooth = (table["mean"] * table["count"] + 20 * gmean) / (table["count"] + 20)
hold["score"] = hold.set_index(["r_b", "n_b"]).index.map(smooth).fillna(gmean).to_numpy()

print("March: empirical smoothed P(y|rec,n) AP@20% =", round(ap20(hold.target.to_numpy(), hold.score, hold.customer_id), 5))
print("\nrate table (rows=recency, cols=n_orders):")
print(smooth.unstack().round(3).to_string())

# добавим med_int в сетку
prior["i_b"] = pd.cut(prior.med_int, [0, 30, 60, 90, 150, 300, 9999])
hold["i_b"] = pd.cut(hold.med_int, [0, 30, 60, 90, 150, 300, 9999])
t2 = prior.groupby(["r_b", "n_b", "i_b"], observed=True).target.agg(["mean", "count"])
s2 = (t2["mean"] * t2["count"] + 30 * gmean) / (t2["count"] + 30)
hold["score2"] = hold.set_index(["r_b", "n_b", "i_b"]).index.map(s2).fillna(hold.score).to_numpy()
print("\nMarch: +median_interval AP@20% =", round(ap20(hold.target.to_numpy(), hold.score2, hold.customer_id), 5))

# оракул-чек: доля покупателей среди top-20% если ранжировать по эмпирической rate
print("\nbase rate:", round(gmean, 4), "| positives in top20%:", int(np.ceil(len(hold)*.2) * 0 + np.nan), )
top_idx = hold.score.rank(ascending=False) <= 5000
print("buyers in top-20% by smoothed rate:", int(hold.target[top_idx].sum()), "/ 5000")
