"""Тест силы interval-rule: score = P(следующая покупка попадёт в окно H после offset a)
на основе эмпирического распределения интервалов клиента.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("/Users/flexonafft/MLKaggleTasks/competitions/chempionats/cu-lamoda-intern/data")
train = pd.read_csv(DATA / "train.csv", parse_dates=["scoring_date"])
orders = pd.read_parquet(DATA / "orders.parquet")
orders["order_date"] = pd.to_datetime(orders["order_date"])
od = orders.groupby(["customer_id", "order_date"]).size().reset_index()[["customer_id", "order_date"]]


def ap20(target, score, ids):
    df = pd.DataFrame({"id": ids.astype(str), "t": target, "s": score}).sort_values(
        ["s", "id"], ascending=[False, True], kind="mergesort")
    k = int(np.ceil(len(df) * 0.2))
    top = df.t.to_numpy()[:k]
    return float((np.cumsum(top) / np.arange(1, k + 1) * top).sum() / max(min(k, int(df.t.sum())), 1))


def interval_rule_score(cust, a, H, use_mean=False):
    """P(next order in [rec_v+a, rec_v+a+H)) по эмпирической CDF интервалов."""
    m = cust.merge(od, on="customer_id", how="left")
    m["rec_v"] = (m.scoring_date - m.order_date).dt.days - 14  # дней от последнего видимого заказа до cutoff
    m = m[m.rec_v >= 0]
    scores = {}
    for cid, g in m.groupby("customer_id"):
        dts = g.order_date.sort_values().drop_duplicates()
        dd = dts.diff().dt.days.dropna().to_numpy()
        rec = g.rec_v.min() + 14  # rec_v relative to scoring (>=14)
        if len(dd) == 0:
            scores[cid] = 0.0
            continue
        center = dd.mean() if use_mean else np.median(dd)
        # вероятность, что следующий заказ в [rec+a, rec+a+H): экспоненциальная近似
        # hazard по эмпирическим интервалам: F(x) = доля интервалов <= x
        lo, hi = rec + a, rec + a + H
        F = lambda x: (dd <= x).mean()
        scores[cid] = max(F(min(hi, center * 3)) - F(lo), 0.0) + 1e-9 / (1 + abs(center - (rec + a + H / 2)))
    return cust.customer_id.map(scores).fillna(0).to_numpy()


for hold_month in ["2022-02-01", "2022-03-01"]:
    hold = train[train.scoring_date == hold_month]
    base = ap20(hold.target.to_numpy(), np.zeros(len(hold)), hold.customer_id.to_numpy())
    print(f"== holdout {hold_month} base AP (const) = {base:.4f}")
    for a in [0, 14]:
        for H in [28, 30, 35, 42, 60]:
            s = interval_rule_score(hold, a, H)
            print(f"  a={a:2d} H={H:2d}: AP@20%={ap20(hold.target.to_numpy(), s, hold.customer_id.to_numpy()):.4f}")
