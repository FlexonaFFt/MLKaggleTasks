"""Чистый reference-пайплайн: RFM + интервалы, без заимствований из ноутбуков.
Цель: проверить, воспроизводится ли ~0.749 локально или базовые фичи дают больше.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

DATA = Path("/Users/flexonafft/MLKaggleTasks/competitions/chempionats/cu-lamoda-intern/data")
GAP = 14
WINDOWS = [30, 60, 90, 180, 365, 730]

train = pd.read_csv(DATA / "train.csv", parse_dates=["scoring_date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["scoring_date"])
orders = pd.read_parquet(DATA / "orders.parquet")
orders["order_date"] = pd.to_datetime(orders["order_date"])

# заказы на уровне корзины
baskets = orders.groupby(["customer_id", "order_id", "order_date"], as_index=False).agg(
    value=("paid_price", "sum"),
    items=("product_id", "size"),
    disc=("discount_percent", "mean"),
    loy=("loyalty_percent", "mean"),
)
baskets = baskets.sort_values(["customer_id", "order_date"]).reset_index(drop=True)


def build_features(cust):
    """cust: df с customer_id, scoring_date. Возвращает фичи на ту же пару."""
    b = cust.merge(baskets, on="customer_id", how="left")
    cutoff = b.scoring_date - pd.Timedelta(days=GAP)
    hist_start = b.scoring_date - pd.DateOffset(years=2) - pd.Timedelta(days=GAP)
    b = b[(b.order_date <= cutoff) & (b.order_date >= hist_start)]

    b["days_ago"] = (b.scoring_date - b.order_date).dt.days
    b["month_ago"] = b.days_ago // 30

    feats = {}

    # recency / frequency / monetary
    feats["recency"] = b.groupby("customer_id").days_ago.min()
    feats["n_orders_all"] = b.groupby("customer_id").size()
    for w in WINDOWS:
        feats[f"n_orders_{w}"] = b[b.days_ago <= w].groupby("customer_id").size()
        feats[f"sum_value_{w}"] = b[b.days_ago <= w].groupby("customer_id").value.sum()
        feats[f"avg_value_{w}"] = b[b.days_ago <= w].groupby("customer_id").value.mean()
    feats["avg_value_all"] = b.groupby("customer_id").value.mean()
    feats["median_value_all"] = b.groupby("customer_id").value.median()
    feats["avg_items"] = b.groupby("customer_id").items.mean()
    feats["avg_disc"] = b.groupby("customer_id").disc.mean()
    feats["avg_loy"] = b.groupby("customer_id").loy.mean()

    out = cust.set_index("customer_id").copy()
    for k, v in feats.items():
        out[k] = v
    out = out.fillna({k: 0 for k in feats})
    out["recency"] = out["recency"].fillna(9999)

    # интервалы между заказами
    ivals = []
    for cid, g in b.groupby("customer_id"):
        dts = g.groupby("order_id").order_date.first().sort_values()
        dd = dts.diff().dt.days.dropna()
        if len(dd) == 0:
            ivals.append((cid, np.nan, np.nan, np.nan, np.nan))
        else:
            ivals.append((cid, dd.iloc[-1], dd.mean(), dd.median(), dd.std()))
    iv = pd.DataFrame(ivals, columns=["customer_id", "last_interval", "mean_interval", "median_interval", "std_interval"]).set_index("customer_id")
    out = out.join(iv)
    out["gap_ratio"] = out.recency / out.median_interval.clip(lower=1)
    out["gap_ratio_mean"] = out.recency / out.mean_interval.clip(lower=1)
    out["interval_cv"] = out.std_interval / out.mean_interval.clip(lower=1)

    # день недели / месяц последнего заказа
    last_date = b.groupby("customer_id").order_date.max()
    out["last_dow"] = last_date.dt.dayofweek
    out["last_month"] = last_date.dt.month
    out["first_days_ago"] = (out.scoring_date - b.groupby("customer_id").order_date.min()).dt.days
    out["active_span"] = out.first_days_ago - out.recency
    return out.reset_index()


def ap20(target, score, ids):
    df = pd.DataFrame({"id": ids.astype(str), "t": target, "s": score}).sort_values(
        ["s", "id"], ascending=[False, True], kind="mergesort")
    k = int(np.ceil(len(df) * 0.2))
    top = df.t.to_numpy()[:k]
    return float((np.cumsum(top) / np.arange(1, k + 1) * top).sum() / max(min(k, int(df.t.sum())), 1))


tr = build_features(train)
te = build_features(test)
cols = [c for c in tr.columns if c not in ("customer_id", "scoring_date", "target")]
print(f"n features: {len(cols)}")

params = dict(objective="binary", learning_rate=.05, num_leaves=47, min_child_samples=85,
              colsample_bytree=.6, subsample=.89, subsample_freq=1, reg_lambda=.66,
              min_split_gain=.43, n_jobs=8, verbose=-1)

for hold_month in ["2022-02-01", "2022-03-01"]:
    prior = tr[tr.scoring_date < hold_month]
    hold = tr[tr.scoring_date == hold_month]
    dtrain = lgb.Dataset(prior[cols], prior.target)
    model = lgb.train(params, dtrain, num_boost_round=2000,
                      valid_sets=[lgb.Dataset(hold[cols], hold.target)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = model.predict(hold[cols], num_iteration=model.best_iteration)
    print(f"holdout {hold_month}: best_iter={model.best_iteration} AP@20%={ap20(hold.target.to_numpy(), pred, hold.customer_id.to_numpy()):.5f}")

fi = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
print(fi.head(15))
