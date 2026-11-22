import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import ndtr
def display(*a, **k): print(*[x.to_string() if hasattr(x,'to_string') else str(x) for x in a] if a else '')

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

KAGGLE_DIR = Path("/kaggle/input/competitions/lamoda-cu-repeat-purchase-2026")
DATA_DIR = KAGGLE_DIR if KAGGLE_DIR.exists() else Path(os.environ.get("LAMODA_DATA_DIR", "data"))
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(os.environ.get("LAMODA_OUT_DIR", "."))

SEED = 42
GAP_DAYS = 14
TARGET_MONTHS = 3
WINDOWS = [7, 14, 30, 60, 90, 180, 365]
T0 = time.time()


def fast_ap(target, score):
    order = np.argsort(-np.asarray(score), kind="mergesort")
    t = np.asarray(target)[order]
    k = int(np.ceil(len(t) * 0.20))
    top = t[:k]
    return float((np.cumsum(top) / np.arange(1, k + 1) * top).sum() / max(min(k, int(t.sum())), 1))

# ====== CELL BREAK ======

train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["scoring_date"])
test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["scoring_date"])
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
orders = pd.read_parquet(DATA_DIR / "orders.parquet")
products = pd.read_parquet(DATA_DIR / "products.parquet")
orders["order_date"] = pd.to_datetime(orders["order_date"])

cust = pd.concat([train.assign(part="train"), test.assign(part="test", target=np.nan)], ignore_index=True)
cust["slice"] = cust["scoring_date"].dt.strftime("%Y-%m-%d")
cust["cutoff"] = cust["scoring_date"] - pd.Timedelta(days=GAP_DAYS)
cust["win_len"] = cust["scoring_date"].map(lambda d: (d + pd.DateOffset(months=TARGET_MONTHS) - d).days)
assert cust["customer_id"].is_unique

lines = orders.merge(products, on="product_id", how="left", validate="m:1").merge(
    cust[["customer_id", "cutoff"]], on="customer_id", how="inner", validate="m:1"
)
lines["d"] = (lines["cutoff"] - lines["order_date"]).dt.days
assert len(lines) == len(orders) and lines["d"].between(0, 732).all()


def clean(name):
    return re.sub(r"\W+", "_", str(name)).strip("_").lower()


def build_features(lines, win_len):
    lines = lines.copy()
    lines["has_disc"] = (lines["discount_percent"] > 0).astype("float32")
    lines["has_loy"] = (lines["loyalty_percent"] > 0).astype("float32")
    lines["weekend"] = (lines["order_date"].dt.dayofweek >= 5).astype("float32")

    # --- уровень заказа, самые свежие заказы идут первыми
    ot = (
        lines.groupby(["customer_id", "order_id"], sort=False)
        .agg(
            d=("d", "first"),
            order_sum=("paid_price", "sum"),
            n_items=("paid_price", "size"),
            disc=("discount_percent", "mean"),
            loy=("loyalty_percent", "max"),
            weekend=("weekend", "first"),
        )
        .reset_index()
        .sort_values(["customer_id", "d", "order_id"])
        .reset_index(drop=True)
    )
    cid = ot["customer_id"]
    f = ot.groupby("customer_id").agg(
        recency=("d", "min"),
        tenure=("d", "max"),
        d_mean=("d", "mean"),
        d_std=("d", "std"),
        n_orders=("order_id", "size"),
        n_items=("n_items", "sum"),
        sum_paid=("order_sum", "sum"),
        order_sum_mean=("order_sum", "mean"),
        order_sum_median=("order_sum", "median"),
        order_sum_max=("order_sum", "max"),
        order_sum_std=("order_sum", "std"),
        items_per_order=("n_items", "mean"),
        items_max=("n_items", "max"),
        order_disc_mean=("disc", "mean"),
        order_loy_mean=("loy", "mean"),
        order_loy_max=("loy", "max"),
        weekend_share=("weekend", "mean"),
    )

    last = ot.drop_duplicates("customer_id").set_index("customer_id")
    first = ot.drop_duplicates("customer_id", keep="last").set_index("customer_id")
    f["last_order_sum"] = last["order_sum"]
    f["last_n_items"] = last["n_items"]
    f["last_disc"] = last["disc"]
    f["last_loy"] = last["loy"]
    f["first_loy"] = first["loy"]
    f["first_order_sum"] = first["order_sum"]
    f["loy_change"] = f["last_loy"] - f["first_loy"]
    f["last_sum_to_mean"] = f["last_order_sum"] / f["order_sum_mean"]

    # --- окна по времени
    for w in WINDOWS:
        m = ot["d"] <= w
        f[f"n_orders_{w}"] = m.groupby(cid).sum()
        f[f"sum_{w}"] = (ot["order_sum"] * m).groupby(cid).sum()
        f[f"n_items_{w}"] = (ot["n_items"] * m).groupby(cid).sum()
    for w in [90, 365]:
        m = ot["d"] <= w
        f[f"loy_max_{w}"] = ot["loy"].where(m).groupby(cid).max()
        f[f"disc_mean_{w}"] = ot["disc"].where(m).groupby(cid).mean()
        f[f"aov_{w}"] = f[f"sum_{w}"] / f[f"n_orders_{w}"].replace(0, np.nan)
    f["n_orders_older_365"] = f["n_orders"] - f["n_orders_365"]

    # --- уровень дня: ритм заказов
    days = ot.groupby(["customer_id", "d"], as_index=False).agg(day_orders=("order_id", "size"), day_sum=("order_sum", "sum"))
    days["gap"] = days.groupby("customer_id")["d"].diff()
    days["rk"] = days.groupby("customer_id").cumcount()
    gd = days.groupby("customer_id")
    f["n_days"] = gd.size()
    f["max_day_orders"] = gd["day_orders"].max()
    f["multi_order_days"] = (days["day_orders"] > 1).groupby(days["customer_id"]).sum()
    f = f.join(gd["gap"].agg(gap_mean="mean", gap_median="median", gap_std="std", gap_min="min", gap_max="max"))
    for k in [1, 2, 4, 9]:
        f[f"d_day_{k + 1}"] = days.loc[days["rk"] == k].set_index("customer_id")["d"]  # давность (k+1)-го с конца дня с заказом
    f["gap_prev"] = days.loc[days["rk"] == 2].set_index("customer_id")["gap"]
    for name, period, limit in [("months_12", 30, 12), ("months_24", 30, 25), ("weeks_13", 7, 13), ("weeks_26", 7, 26)]:
        b = days.assign(b=days["d"] // period)
        f[f"active_{name}"] = b.loc[b["b"] < limit].drop_duplicates(["customer_id", "b"]).groupby("customer_id").size()
    f[[c for c in f.columns if c.startswith("active_")]] = f[[c for c in f.columns if c.startswith("active_")]].fillna(0)

    # --- попадание следующего заказа в целевое окно
    # Время считаем в днях после отсечения: окно — это (GAP_DAYS, GAP_DAYS + win_len].
    r = f["recency"]
    w0, w1 = GAP_DAYS, GAP_DAYS + win_len.reindex(f.index)
    dc = days["customer_id"]
    g, dr = days["gap"], dc.map(r)
    lo, hi = dr + GAP_DAYS, dr + GAP_DAYS + dc.map(win_len)
    n_gaps = g.notna().groupby(dc).sum()
    alive = (g > dr).groupby(dc).sum()  # сколько собственных интервалов клиента длиннее текущей паузы
    f["own_gaps_gt_recency"] = alive
    f["own_share_gaps_gt_recency"] = alive / n_gaps.replace(0, np.nan)
    f["own_p_hit"] = ((g > lo) & (g <= hi)).groupby(dc).sum() / alive.replace(0, np.nan)
    f["own_p_hidden"] = ((g > dr) & (g <= lo)).groupby(dc).sum() / alive.replace(0, np.nan)
    f["own_p_late"] = (g > hi).groupby(dc).sum() / alive.replace(0, np.nan)
    f["own_share_gaps_le_window"] = (g <= dc.map(win_len)).groupby(dc).sum() / n_gaps.replace(0, np.nan)

    wgt = 0.7 ** (days["rk"] - 1).clip(lower=0)
    f["gap_ewm"] = (g * wgt).groupby(dc).sum() / wgt.where(g.notna()).groupby(dc).sum()
    f["gap_last3"] = g.where(days["rk"].between(1, 3)).groupby(dc).mean()
    f["gap_last"] = days.loc[days["rk"] == 1].set_index("customer_id")["gap"]
    f["gap_last_to_median"] = f["gap_last"] / f["gap_median"]
    for name in ["gap_last", "gap_last3", "gap_ewm", "gap_median", "gap_mean", "gap_min", "gap_max"]:
        short = name.replace("gap_", "")
        gg = f[name].where(f[name] > 0)
        t = gg - r  # ожидаемый день следующего заказа; отрицательный — клиент «опаздывает»
        f[f"tnext_{short}"] = t
        f[f"tnext_{short}_dist"] = (w0 - t).clip(lower=0) + (t - w1).clip(lower=0)  # расстояние до окна, 0 — внутри
        if name in ["gap_last", "gap_ewm", "gap_median", "gap_mean"]:
            f[f"cycles_{short}"] = np.floor((r + w1) / gg) - np.floor((r + w0) / gg)  # сколько раз период укладывается в окно
            f[f"tmod_{short}"] = gg - (r % gg)  # до ближайшего кратного периоду дня, если клиент пропускал циклы

    # --- окно в единицах собственного периода и сглаженное попадание
    for name in ["gap_mean", "gap_median", "gap_last", "gap_last3"]:
        short = name.replace("gap_", "")
        gg = f[name].where(f[name] > 0)
        t = gg - r
        f[f"ulo_{short}"] = (r + w0) / gg  # следующий заказ попадает в окно, если ulo < 1 <= uhi
        f[f"uhi_{short}"] = (r + w1) / gg
        for sd in (10, 20):  # день следующего заказа известен с точностью ± sd дней
            f[f"phit{sd}_{short}"] = ndtr((w1 - t) / sd) - ndtr((w0 - t) / sd)
        f[f"phitrel_{short}"] = ndtr((w1 - t) / (0.25 * gg)) - ndtr((w0 - t) / (0.25 * gg))

    # --- дни с заказами по 30-дневным интервалам и в том же окне год и два назад
    mb = pd.crosstab(dc, (days["d"] // 30).clip(upper=24)).reindex(columns=range(24), fill_value=0)
    mb.columns = [f"mb_{i:02d}" for i in mb.columns]
    f = f.join(mb)
    dl = dc.map(win_len)
    for years, tag in [(1, "ya"), (2, "ya2")]:
        start, end = 365 * years - GAP_DAYS - dl, 365 * years - GAP_DAYS - 1
        f[f"{tag}_days"] = ((days["d"] >= start) & (days["d"] <= end)).groupby(dc).sum()
        f[f"{tag}_days_wide"] = ((days["d"] >= start - 30) & (days["d"] <= end + 30)).groupby(dc).sum()

    # --- производные
    f["formula"] = f["n_orders"] / (f["recency"] + 30)
    f["days_formula"] = f["n_days"] / (f["recency"] + 30)
    f["orders_per_30d"] = f["n_orders"] / (f["tenure"] + 30) * 30
    f["recency_to_gap_mean"] = f["recency"] / f["gap_mean"]
    f["recency_to_gap_median"] = f["recency"] / f["gap_median"]
    f["recency_minus_gap_max"] = f["recency"] - f["gap_max"]
    f["gap_cv"] = f["gap_std"] / f["gap_mean"]
    f["recency_to_tenure"] = f["recency"] / (f["tenure"] + 1)
    f["share_orders_30"] = f["n_orders_30"] / f["n_orders"]
    f["share_orders_90"] = f["n_orders_90"] / f["n_orders"]
    f["share_orders_365"] = f["n_orders_365"] / f["n_orders"]
    f["trend_90_vs_365"] = f["n_orders_90"] / (f["n_orders_365"] - f["n_orders_90"] + 1)
    f["trend_30_vs_90"] = f["n_orders_30"] / (f["n_orders_90"] - f["n_orders_30"] + 1)
    f["share_sum_90"] = f["sum_90"] / f["sum_paid"].replace(0, np.nan)

    # --- уровень позиции: цены, скидки, лояльность, состав покупок
    gl = lines.groupby("customer_id")
    f = f.join(
        gl.agg(
            price_mean=("paid_price", "mean"),
            price_median=("paid_price", "median"),
            price_max=("paid_price", "max"),
            price_min=("paid_price", "min"),
            disc_mean=("discount_percent", "mean"),
            disc_max=("discount_percent", "max"),
            disc_share=("has_disc", "mean"),
            loy_mean=("loyalty_percent", "mean"),
            loy_share=("has_loy", "mean"),
            n_loy_values=("loyalty_percent", "nunique"),
            n_products=("product_id", "nunique"),
            n_brands=("brand", "nunique"),
            n_cat=("category", "nunique"),
            n_cat2=("category2", "nunique"),
            n_gender=("gender_product", "nunique"),
        )
    )
    f["repeat_product_share"] = 1 - f["n_products"] / f["n_items"]
    f["brands_per_item"] = f["n_brands"] / f["n_items"]
    f["cat2_per_item"] = f["n_cat2"] / f["n_items"]

    for col, prefix in [("category", "cat"), ("gender_product", "gender")]:
        share = pd.crosstab(lines["customer_id"], lines[col], normalize="index")
        share.columns = [f"{prefix}_share_{clean(c)}" for c in share.columns]
        f = f.join(share)
        recent = lines.loc[lines["d"] <= 90]
        share = pd.crosstab(recent["customer_id"], recent[col], normalize="index")
        share.columns = [f"{prefix}_share90_{clean(c)}" for c in share.columns]
        f = f.join(share)

    for col in ["category2", "brand"]:
        top = (
            lines.groupby(["customer_id", col], observed=True).size().rename("n").reset_index()
            .sort_values(["customer_id", "n", col], ascending=[True, False, True])
            .drop_duplicates("customer_id").set_index("customer_id")
        )
        f[f"top_{col}_share"] = top["n"] / f["n_items"]
        f[f"top_{col}"] = top[col].astype(str)

    # --- ритм отдельно по категориям, подкатегориям и брендам
    for col in ["category", "category2", "brand"]:
        st = (
            lines[["customer_id", col, "d"]].drop_duplicates()
            .groupby(["customer_id", col], observed=True)["d"].agg(n="size", last="min", first="max")
        )
        st = st[st["n"] >= 2].reset_index()
        st["gap"] = (st["first"] - st["last"]) / (st["n"] - 1)
        st["t"] = st["gap"] - st["last"]
        hi = GAP_DAYS + st["customer_id"].map(win_len)
        st["dist"] = (GAP_DAYS - st["t"]).clip(lower=0) + (st["t"] - hi).clip(lower=0)
        st["hit"] = (st["dist"] == 0).astype(float)
        gs = st.groupby("customer_id")
        f[f"ser_{col}_n"] = gs.size()
        f[f"ser_{col}_hit"] = gs["hit"].sum()
        f[[f"ser_{col}_n", f"ser_{col}_hit"]] = f[[f"ser_{col}_n", f"ser_{col}_hit"]].fillna(0)
        f[f"ser_{col}_hit_share"] = gs["hit"].mean()
        f[f"ser_{col}_min_dist"] = gs["dist"].min()
        best = st.sort_values(["customer_id", "dist", "last"]).drop_duplicates("customer_id").set_index("customer_id")
        f[f"ser_{col}_best_t"] = best["t"]
        f[f"ser_{col}_best_gap"] = best["gap"]

    return f.replace([np.inf, -np.inf], np.nan)


feats = build_features(lines, cust.set_index("customer_id")["win_len"])
data = cust.merge(feats, left_on="customer_id", right_index=True, how="left", validate="1:1")
assert data["recency"].notna().all() and len(data) == len(cust)
data["top_category2"] = data["top_category2"].astype("category")
data["top_brand"] = data["top_brand"].astype(object).where(data["top_brand"].notna(), "none").astype(str)

# ====== CELL BREAK ======

cid = "customer_id"
info = cust.set_index(cid)
LO = pd.Series(float(GAP_DAYS), index=info.index)
HI = (GAP_DAYS + info["win_len"]).astype(float)
EST = ["mean", "median", "min", "max", "q25", "q75", "last", "last2", "last3", "ewm", "trim"]
HIST = {"вся": 10_000, "год": 365, "полгода": 182}


def month_idx(s):
    return s.dt.year * 12 + s.dt.month


def estimators(E):
    E = E.sort_values([cid, "t"]).reset_index(drop=True)
    g = E.groupby(cid)["t"]
    E["gap"] = g.diff()
    E["rk"] = E.groupby(cid).cumcount()
    gp = E[E["gap"].notna()]
    gg = gp.groupby(cid)["gap"]
    out = pd.DataFrame({"r": g.min(), "n": g.size()})
    out["mean"] = (g.max() - g.min()) / (out["n"] - 1).replace(0, np.nan)
    out["median"] = gg.median()
    out["min"] = gg.min()
    out["max"] = gg.max()
    out["q25"] = gg.quantile(0.25)
    out["q75"] = gg.quantile(0.75)
    out["last"] = gp[gp["rk"] == 1].set_index(cid)["gap"]
    out["last2"] = gp[gp["rk"] <= 2].groupby(cid)["gap"].mean()
    out["last3"] = gp[gp["rk"] <= 3].groupby(cid)["gap"].mean()
    w = 0.7 ** (gp["rk"] - 1)
    out["ewm"] = (gp["gap"] * w).groupby(gp[cid]).sum() / w.groupby(gp[cid]).sum()
    out["trim"] = (gg.sum() - gg.max()) / (gg.count() - 1).replace(0, np.nan)
    return out


def unit_feats(E, prefix):
    # «ожидаемый момент следующего заказа» по разным оценкам периода; E — (customer_id, d), d — дней до отсечения
    out = {}
    E = E.assign(t=E["d"].astype(float))
    for hname, H in HIST.items():
        est = estimators(E[E["d"] <= H][[cid, "t"]]).reindex(info.index)
        r = est["r"]
        key = f"{prefix}|{hname}"
        out[f"{key}|n"] = est["n"].fillna(0)
        out[f"{key}|r"] = r
        for e in EST:
            P = est[e].where(est[e] > 0)
            tn, tm = P - r, P - np.mod(r, P)
            pre = f"{key}|{e}"
            out[f"{pre}|t"] = tn
            out[f"{pre}|dist"] = (LO - tn).clip(lower=0) + (tn - HI).clip(lower=0)
            out[f"{pre}|in"] = ((tn > LO) & (tn <= HI)).astype("float32").where(P.notna())
            out[f"{pre}|tm_in"] = ((tm > LO) & (tm <= HI)).astype("float32").where(P.notna())
            out[f"{pre}|k_in"] = np.floor((r + HI) / P) - np.floor((r + LO) / P)
    return out


o = (
    lines.groupby([cid, "order_id"], sort=False)
    .agg(d=("d", "first"), date=("order_date", "first"), n_items=("paid_price", "size"))
    .reset_index()
    .sort_values([cid, "d", "order_id"])
    .reset_index(drop=True)
)
new = unit_feats(o[[cid, "d"]], "заказ")

# структура интервалов между заказами
o["gap"] = o.groupby(cid)["d"].diff()
gpo = o[o["gap"].notna()]
for name, cond in [("0", gpo["gap"] == 0), ("le1", gpo["gap"] <= 1), ("le3", gpo["gap"] <= 3), ("le7", gpo["gap"] <= 7), ("le14", gpo["gap"] <= 14),
                   ("le30", gpo["gap"] <= 30), ("gt60", gpo["gap"] > 60), ("gt90", gpo["gap"] > 90), ("gt180", gpo["gap"] > 180)]:
    cf = cond.astype(float)
    new[f"интервалы|доля {name}"] = cf.groupby(gpo[cid]).mean()
    new[f"интервалы|доля {name} за год"] = cf.where(gpo["d"] <= 365).groupby(gpo[cid]).mean()
    new[f"интервалы|число {name} за год"] = cf.where(gpo["d"] <= 365).groupby(gpo[cid]).sum()
dayc = o.groupby([cid, "d"]).agg(k=("order_id", "size"), items=("n_items", "sum")).reset_index()
for w in (90, 365):
    new[f"интервалы|дней с 2+ заказами за {w}"] = ((dayc["k"] >= 2) & (dayc["d"] <= w)).groupby(dayc[cid]).sum()
last_day = dayc.sort_values([cid, "d"]).drop_duplicates(cid).set_index(cid)
new["интервалы|заказов в последний день"] = last_day["k"]
new["интервалы|позиций в последний день"] = last_day["items"]
new["интервалы|давность последнего дня с 2+ заказами"] = dayc[dayc["k"] >= 2].groupby(cid)["d"].min()
r_o = o.groupby(cid)["d"].min()
qs = gpo.groupby(cid)["gap"].quantile([0.25, 0.5, 0.75, 0.9]).unstack()
for q in qs.columns:
    new[f"интервалы|давность / квантиль {q}"] = r_o / qs[q].replace(0, np.nan)
    new[f"интервалы|давность - квантиль {q}"] = r_o - qs[q]
new["интервалы|разброс"] = gpo.groupby(cid)["gap"].std() / gpo.groupby(cid)["gap"].mean().replace(0, np.nan)
pos = gpo[gpo["gap"] > 0].groupby(cid)["gap"]
new["интервалы|средний ненулевой"] = pos.mean()
new["интервалы|давность - средний ненулевой"] = r_o - pos.mean()

# заказы и позиции по непересекающимся интервалам до отсечения
BINS = [0, 7, 14, 30, 60, 90, 120, 180, 270, 365, 450, 540, 733]
labels = [f"{BINS[i]}-{BINS[i + 1]}" for i in range(len(BINS) - 1)]
t_o = pd.crosstab(o[cid], pd.cut(o["d"], BINS, right=False, labels=labels)).reindex(columns=labels, fill_value=0)
t_l = pd.crosstab(lines[cid], pd.cut(lines["d"], BINS, right=False, labels=labels)).reindex(columns=labels, fill_value=0)
for lab in labels:
    new[f"по интервалам|заказов {lab}"] = t_o[lab]
    new[f"по интервалам|позиций {lab}"] = t_l[lab]
    new[f"по интервалам|доля заказов {lab}"] = t_o[lab] / t_o.sum(axis=1)

# календарь
mi = month_idx(o["date"])
rel = o[cid].map(month_idx(info["scoring_date"])) - mi
last_o = o.drop_duplicates(cid).set_index(cid)
new["календарь|день месяца последнего заказа"] = last_o["date"].dt.day
new["календарь|день недели последнего заказа"] = last_o["date"].dt.dayofweek
new["календарь|месяц последнего заказа"] = last_o["date"].dt.month
new["календарь|доля заказов 1–10 числа"] = (o["date"].dt.day <= 10).groupby(o[cid]).mean()
new["календарь|доля заказов 21+ числа"] = (o["date"].dt.day >= 21).groupby(o[cid]).mean()
new["календарь|заказов в месяцы окна год назад"] = rel.between(10, 12).groupby(o[cid]).sum()
new["календарь|заказов в месяцы окна два года назад"] = rel.between(22, 24).groupby(o[cid]).sum()
new["календарь|заказов в 3 месяца перед окном год назад"] = rel.between(13, 15).groupby(o[cid]).sum()
new["календарь|заказов в 3 месяца после окна год назад"] = rel.between(7, 9).groupby(o[cid]).sum()
new["календарь|активных месяцев"] = mi.groupby(o[cid]).nunique()
new["календарь|активных месяцев за год"] = mi.where(rel <= 12).groupby(o[cid]).nunique()

NEW = pd.DataFrame(new).reindex(info.index).astype("float32")
NAMES = {f"n{i}": c for i, c in enumerate(NEW.columns)}       # короткие имена для моделей -> понятные имена
NEW.columns = list(NAMES)
data = data.join(NEW, on=cid)
BASE = [c for c in feats.columns if c != "top_brand"]
print(f"новых признаков: {NEW.shape[1]} | {(time.time() - T0) / 60:.1f} мин")

# ====== CELL BREAK ======

# Фиксируем все признаки, рассчитанные только из доступной истории; не ранжируем их по таргету.
FEATURES = BASE + list(NAMES)


# ====== CELL BREAK ======

IDX = info.index
L = lines
OT = (L.groupby([cid, "order_id"]).agg(d=("d", "first"), s=("paid_price", "sum"), n=("paid_price", "size"),
                                       disc=("discount_percent", "mean"), loy=("loyalty_percent", "max"))
      .reset_index().sort_values([cid, "d", "order_id"]).reset_index(drop=True))
OT["k"] = OT.groupby(cid).cumcount()                 # 0 — самый свежий заказ
OT["gap"] = OT.groupby(cid)["d"].diff()              # пауза до соседнего более свежего заказа
R = OT.groupby(cid)["d"].min().reindex(IDX)
BANK = {}


def put(group, name, s):
    BANK.setdefault(group, {})[name] = pd.Series(s).reindex(IDX)


def ratio(a, b):
    return a / b.replace(0, np.nan)


# окна по времени и тренды между соседними окнами
W = [30, 60, 90, 120, 180, 270, 365, 540, 730]
cnt, summ = {}, {}
for w in W:
    m = OT["d"] < w
    cnt[w] = m.groupby(OT[cid]).sum().reindex(IDX).fillna(0)
    summ[w] = OT["s"].where(m, 0).groupby(OT[cid]).sum().reindex(IDX).fillna(0)
    put("окна", f"заказов <{w}", cnt[w])
    put("окна", f"сумма <{w}", summ[w])
    put("окна", f"позиций <{w}", OT["n"].where(m, 0).groupby(OT[cid]).sum())
    put("окна", f"скидка <{w}", OT["disc"].where(m).groupby(OT[cid]).mean())
for a, b in zip(W[:-1], W[1:]):
    put("окна", f"тренд заказов {a}/{b}", ratio(cnt[a], cnt[b] - cnt[a] + 1))
    put("окна", f"тренд суммы {a}/{b}", ratio(summ[a], summ[b] - summ[a] + 1))
    put("окна", f"доля заказов {a} из {b}", ratio(cnt[a], cnt[b]))

# последние k заказов
for k in [2, 3, 5, 10]:
    sub = OT[OT["k"] < k]
    g = sub.groupby(cid)
    c = g.size()
    span = g["d"].max() - g["d"].min()
    put("последние k", f"сумма ср {k}", g["s"].mean())
    put("последние k", f"позиций ср {k}", g["n"].mean())
    put("последние k", f"скидка ср {k}", g["disc"].mean())
    put("последние k", f"размах {k}", span.where(c == k))
    put("последние k", f"пауза ср {k}", (span / (c - 1).replace(0, np.nan)).where(c == k))
    put("последние k", f"давность {k}-го", g["d"].max().where(c == k))
    put("последние k", f"давность / пауза {k}", ratio(R, (span / (c - 1).replace(0, np.nan)).where(c == k).reindex(IDX)))

# статистики пауз по окнам
gp = OT[OT["gap"].notna()]
for H in [90, 180, 365, 730]:
    g = gp[gp["d"] < H].groupby(cid)["gap"]
    mean, med = g.mean().reindex(IDX), g.median().reindex(IDX)
    put("паузы", f"ср <{H}", mean)
    put("паузы", f"медиана <{H}", med)
    put("паузы", f"разброс <{H}", g.std())
    put("паузы", f"мин <{H}", g.min())
    put("паузы", f"макс <{H}", g.max())
    put("паузы", f"асимметрия <{H}", g.skew())
    for q in [7, 30, 90]:
        put("паузы", f"доля ≤{q} <{H}", (gp["gap"] <= q).where(gp["d"] < H).groupby(gp[cid]).mean())
    put("паузы", f"давность - ср <{H}", R - mean)
    put("паузы", f"давность / медиана <{H}", ratio(R, med))
    put("паузы", f"макс - давность <{H}", g.max().reindex(IDX) - R)

# повторы товаров, брендов и подкатегорий
L2 = L[[cid, "d", "product_id", "brand", "category2", "category"]].copy()
L2["last"] = L2["d"].eq(L2[cid].map(R))
for key in ["product_id", "brand", "category2", "category"]:
    u = L2.dropna(subset=[key]).drop_duplicates([cid, "d", key])
    days_per = u.groupby([cid, key]).size()
    put("повторы", f"{key}: доля повторных", (days_per >= 2).groupby(level=0).mean())
    put("повторы", f"{key}: макс дней", days_per.groupby(level=0).max())
    before = u[~u["last"]][[cid, key]].drop_duplicates().assign(seen=1)
    lastk = u[u["last"]][[cid, key]].drop_duplicates().merge(before, on=[cid, key], how="left")
    put("повторы", f"{key}: последний заказ уже покупал", lastk["seen"].fillna(0).groupby(lastk[cid]).mean())

# денежные тренды
for a, b in [(90, 180), (180, 365), (365, 730)]:
    put("деньги", f"сумма {a} / {a}-{b}", ratio(summ[a], summ[b] - summ[a]))
put("деньги", "средний чек 365 / 730", ratio(ratio(summ[365], cnt[365]), ratio(summ[730], cnt[730])))
put("деньги", "чек последнего / средний", ratio(OT[OT["k"] == 0].set_index(cid)["s"].reindex(IDX), OT.groupby(cid)["s"].mean().reindex(IDX)))
put("деньги", "лояльность сейчас - год назад", OT[OT["k"] == 0].set_index(cid)["loy"].reindex(IDX)
    - OT[OT["d"] >= 300].groupby(cid)["loy"].first().reindex(IDX))
put("деньги", "скидка 90 - скидка 365", OT["disc"].where(OT["d"] < 90).groupby(OT[cid]).mean().reindex(IDX)
    - OT["disc"].where(OT["d"] < 365).groupby(OT[cid]).mean().reindex(IDX))

# давность и частота по каждой категории
for c in L["category"].dropna().unique():
    s = L[L["category"] == c]
    dd = s.drop_duplicates([cid, "d"])
    put("категории", f"{c}: давность", dd.groupby(cid)["d"].min())
    put("категории", f"{c}: дней <365", (dd["d"] < 365).groupby(dd[cid]).sum())
    put("категории", f"{c}: доля позиций <180", ratio((s["d"] < 180).groupby(s[cid]).sum().reindex(IDX),
                                                       (L["d"] < 180).groupby(L[cid]).sum().reindex(IDX)))

# бренды
b365 = L[L["d"] < 365]
put("бренды", "брендов <365", b365.groupby(cid)["brand"].nunique())
put("бренды", "брендов <90", L[L["d"] < 90].groupby(cid)["brand"].nunique())
put("бренды", "брендов на заказ", ratio(L.groupby(cid)["brand"].nunique().reindex(IDX), OT.groupby(cid).size().reindex(IDX)))
topb = L.groupby([cid, "brand"]).size()
put("бренды", "доля главного бренда", topb.groupby(level=0).max() / topb.groupby(level=0).sum())
put("бренды", "подкатегорий <365", b365.groupby(cid)["category2"].nunique())

GROUPS, BANK_NAMES = {}, {}
for gname, cols in BANK.items():
    df = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).astype("float32")
    short = [f"b_{len(BANK_NAMES) + i}" for i in range(df.shape[1])]
    BANK_NAMES.update(dict(zip(short, df.columns)))
    df.columns = short
    data = data.join(df, on=cid)
    GROUPS[gname] = short
print("групп:", len(GROUPS), "| признаков в банке:", len(BANK_NAMES), {g: len(c) for g, c in GROUPS.items()}, f"| {(time.time() - T0) / 60:.1f} мин")

# ====== CELL BREAK ======

BANK = {}
for col in ["category", "category2"]:
    dd = L[[cid, col, "d"]].dropna().drop_duplicates().sort_values([cid, col, "d"])
    dd["gap"] = dd.groupby([cid, col])["d"].diff()
    st = dd.groupby([cid, col]).agg(last=("d", "min"), n=("d", "size"), med=("gap", "median"), mx=("gap", "max")).reset_index()
    st = st[st["n"] >= 2]
    st["r_med"] = st["last"] / st["med"].replace(0, np.nan)
    st["med_r"] = st["med"] - st["last"]
    st["mx_r"] = st["mx"] - st["last"]
    g = st.groupby(cid)
    put("паузы в категориях", f"{col}: мин давность / медиана", g["r_med"].min())
    put("паузы в категориях", f"{col}: ср давность / медиана", g["r_med"].mean())
    put("паузы в категориях", f"{col}: медиана - давность, макс", g["med_r"].max())
    put("паузы в категориях", f"{col}: медиана - давность, мин |.|", st.assign(a=st["med_r"].abs()).groupby(cid)["a"].min())
    put("паузы в категориях", f"{col}: макс пауза - давность, макс", g["mx_r"].max())
    put("паузы в категориях", f"{col}: серий 3+ дней", (st["n"] >= 3).groupby(st[cid]).sum())
    top = st.sort_values([cid, "n", "last"], ascending=[True, False, True]).drop_duplicates(cid).set_index(cid)
    put("паузы в категориях", f"{col} главная: медиана", top["med"])
    put("паузы в категориях", f"{col} главная: давность", top["last"])
    put("паузы в категориях", f"{col} главная: давность / медиана", top["r_med"])

for gname, cols in BANK.items():
    df = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).astype("float32")
    short = [f"b_{len(BANK_NAMES) + i}" for i in range(df.shape[1])]
    BANK_NAMES.update(dict(zip(short, df.columns)))
    df.columns = short
    data = data.join(df, on=cid)
    GROUPS[gname] = short
FEATURES_V2 = FEATURES
FEATURES = FEATURES_V2 + GROUPS["паузы"] + GROUPS["категории"] + GROUPS["паузы в категориях"]
print("признаков v2:", len(FEATURES_V2), "| с банком:", len(FEATURES), f"| {(time.time() - T0) / 60:.1f} мин")

# ====== CELL BREAK ======

FEATURES_V4 = FEATURES
BANK = {}
HI_ = GAP_DAYS + info["win_len"].astype(float)       # целевое окно в днях после отсечения: (GAP_DAYS, HI_]


def cycles(col, src=None):
    # по каждой паре (клиент, значение col): давность последней покупки, первой, число дней, медиана и максимум паузы
    src = L if src is None else src
    dd = src[[cid, col, "d"]].dropna().drop_duplicates().sort_values([cid, col, "d"])
    dd["gap"] = dd.groupby([cid, col])["d"].diff()
    st = dd.groupby([cid, col]).agg(last=("d", "min"), first=("d", "max"), n=("d", "size"), med=("gap", "median"),
                                     mx=("gap", "max")).reset_index()
    st = st[st["n"] >= 2].copy()
    st["r_med"] = st["last"] / st["med"].replace(0, np.nan)
    st["t_med"] = st["med"] - st["last"]                                  # ожидаемая следующая покупка по медиане
    st["t_mean"] = (st["first"] - st["last"]) / (st["n"] - 1) - st["last"]  # то же по средней паузе
    return st


# паузы отдельно по каждой категории и полу товара
for col in ["category", "gender_product"]:
    st = cycles(col)
    for c in L[col].dropna().unique():
        s = st[st[col] == c].set_index(cid)
        put("паузы по каждой категории", f"{col} {c}: медиана", s["med"])
        put("паузы по каждой категории", f"{col} {c}: давность / медиана", s["r_med"])
        put("паузы по каждой категории", f"{col} {c}: медиана - давность", s["t_med"])
        put("паузы по каждой категории", f"{col} {c}: средняя - давность", s["t_mean"])
        put("паузы по каждой категории", f"{col} {c}: макс пауза", s["mx"])

# попадает ли ожидаемая покупка по категориям в целевое окно
for col in ["category", "category2", "brand"]:
    st = cycles(col)
    if col == "category":
        overdue_share = st["last"].gt(st["med"]).groupby(st[cid]).mean()
    hi = st[cid].map(HI_)
    mid = (GAP_DAYS + hi) / 2
    for t in ["t_med", "t_mean"]:
        inn = ((st[t] > GAP_DAYS) & (st[t] <= hi)).astype(float)
        dist = (GAP_DAYS - st[t]).clip(lower=0) + (st[t] - hi).clip(lower=0)
        put("окно по категориям", f"{col} {t}: в окне", inn.groupby(st[cid]).sum())
        put("окно по категориям", f"{col} {t}: доля в окне", inn.groupby(st[cid]).mean())
        put("окно по категориям", f"{col} {t}: доля в окне, вес по дням", (inn * st["n"]).groupby(st[cid]).sum() / st["n"].groupby(st[cid]).sum())
        put("окно по категориям", f"{col} {t}: мин расстояние", dist.groupby(st[cid]).min())
        put("окно по категориям", f"{col} {t}: мин |t - середина окна|", (st[t] - mid).abs().groupby(st[cid]).min())
    put("окно по категориям", f"{col}: t_mean в (35, 85]", st["t_mean"].between(35, 85, inclusive="right").groupby(st[cid]).sum())

for gname, cols in BANK.items():
    df = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).astype("float32")
    short = [f"b_{len(BANK_NAMES) + i}" for i in range(df.shape[1])]
    BANK_NAMES.update(dict(zip(short, df.columns)))
    df.columns = short
    data = data.join(df, on=cid)
    GROUPS[gname] = short
FEATURES = FEATURES_V4 + GROUPS["паузы по каждой категории"] + GROUPS["окно по категориям"]
print("признаков v4:", len(FEATURES_V4), "| v5:", len(FEATURES), f"| {(time.time() - T0) / 60:.1f} мин")

# ====== CELL BREAK ======

daily = lines.groupby([cid, "order_date", "category"], observed=True).size().unstack(fill_value=0).sort_index()
daily = daily.div(daily.sum(axis=1), axis=0)
recent = daily.groupby(level=0).tail(3)
earlier = daily.drop(recent.index)
drift = .5 * (recent.groupby(level=0).mean() - earlier.groupby(level=0).mean()).abs().sum(axis=1, min_count=1)
drift = drift.where(daily.groupby(level=0).size().ge(6))
assert drift.dropna().between(0, 1 + 1e-12).all()
data["basket_drift"] = data[cid].map(drift.clip(0, 1))

history_rows = []
for customer, baskets in daily.groupby(level=0, sort=False):
    ages = (info.at[customer, "cutoff"] - pd.to_datetime(baskets.index.get_level_values(1))).days.to_numpy()
    gaps = -np.diff(ages)
    vectors = baskets.to_numpy()
    recency = ages[-1]
    values = [np.nan] * 6
    if len(gaps):
        weights = np.minimum(vectors[:-1], vectors[-1]).sum(axis=1)**2
        survived = gaps > recency
        in_window = (gaps > recency + GAP_DAYS) & (gaps <= recency + GAP_DAYS + info.at[customer, "win_len"])
        values[:3] = [weights.sum(), weights[survived].sum(), weights[in_window].sum()]
        if weights.sum() > 0:
            average_gap = np.average(gaps, weights=weights)
            values[3:5] = [average_gap - gaps.mean(), recency - average_gap]
        if weights[survived].sum() > 0:
            values[5] = np.clip(values[2] / values[1], 0, 1)
    history_rows.append((customer, *values))
basket_columns = [f"basket_wait_{i}" for i in range(6)]
basket = pd.DataFrame(history_rows, columns=[cid, *basket_columns]).set_index(cid)
assert basket[basket_columns[-1]].dropna().between(0, 1).all()
data = data.join(basket, on=cid)
new_columns = ["basket_drift", *basket_columns]

# ====== CELL BREAK ======

from sklearn.model_selection import train_test_split
from time import perf_counter

tr = data[data.part.eq("train")].reset_index(drop=True)
te = data[data.part.eq("test")].set_index(cid).loc[sample[cid]].reset_index()
y = tr.target.astype(int).to_numpy()
days = tr["slice"].to_numpy()

GRID = {
    "g0_base": dict(objective="binary", learning_rate=.05, subsample_freq=1, num_leaves=47,
                    min_child_samples=85, colsample_bytree=.6, subsample=.89, reg_lambda=.66,
                    min_split_gain=.43, n_jobs=4, verbose=-1),
    "g1_wide": dict(objective="binary", learning_rate=.03, subsample_freq=1, num_leaves=127,
                    min_child_samples=40, colsample_bytree=.5, subsample=.8, reg_lambda=1.0,
                    min_split_gain=0.0, n_jobs=4, verbose=-1),
    "g2_shallow": dict(objective="binary", learning_rate=.05, subsample_freq=1, num_leaves=31,
                       min_child_samples=150, colsample_bytree=.8, subsample=.9, reg_lambda=.5,
                       min_split_gain=0.0, n_jobs=4, verbose=-1),
    "g3_deep": dict(objective="binary", learning_rate=.02, subsample_freq=1, num_leaves=255,
                    min_child_samples=20, colsample_bytree=.4, subsample=.7, reg_lambda=2.0,
                    min_split_gain=0.0, max_bin=511, n_jobs=4, verbose=-1),
    "g4_xt": dict(objective="binary", learning_rate=.05, subsample_freq=1, num_leaves=63,
                  min_child_samples=60, colsample_bytree=.7, subsample=.85, reg_lambda=.3,
                  min_split_gain=0.0, extra_trees=True, n_jobs=4, verbose=-1),
}
FEAT = list(FEATURES)
SEEDS_SEARCH = (42, 43)
print(f"признаков: {len(FEAT)}")


def percentile(p):
    return pd.Series(p).rank(pct=True).to_numpy()


def metrics(target, pred, ids):
    order = np.lexsort((ids.astype(str), -pred))
    k = int(np.ceil(len(target) * .2))
    return fast_ap(target[order], pred[order]), int(target[order[:k]].sum())


records = []
rounds = {}
for month in np.unique(days)[1:]:
    prior, holdout = np.flatnonzero(days < month), np.flatnonzero(days == month)
    for name, params in GRID.items():
        t0 = perf_counter()
        preds = []
        rounds[month, name] = []
        for seed in SEEDS_SEARCH:
            fit_idx, stop_idx = train_test_split(prior, test_size=.1, random_state=seed, stratify=y[prior])
            model = lgb.LGBMClassifier(**params, n_estimators=2500, random_state=seed)
            model.fit(tr[FEAT].iloc[fit_idx], y[fit_idx], eval_set=[(tr[FEAT].iloc[stop_idx], y[stop_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            rounds[month, name].append(model.best_iteration_)
            preds.append(percentile(model.predict_proba(tr[FEAT].iloc[holdout])[:, 1]))
        ap, buyers = metrics(y[holdout], np.mean(preds, axis=0), tr[cid].to_numpy()[holdout])
        records.append({"month": month, "config": name, "ap20": ap, "buyers_top20": buyers,
                        "sec": perf_counter() - t0})
        print(month, name, f"AP@20%={ap:.5f}", f"{records[-1]['sec']:.0f} s", flush=True)

grid_results = pd.DataFrame(records)
pivot = grid_results.pivot_table(index="config", columns="month", values="ap20", aggfunc="mean")
pivot["min_month"] = pivot.min(axis=1)
pivot = pivot.sort_values("min_month", ascending=False)
print("\nСредний AP@20% по месяцам (min_month — защита от подгонки):")
display(pivot)
grid_results.to_csv(OUT_DIR / "grid_v9.csv", index=False)
best_two = list(pivot.index[:2])
print("Финал: конфиги", best_two)

# ====== CELL BREAK ======

FINAL_SEEDS = tuple(range(42, 50))  # 8 seed
ROUND_SCALE = 1.2
ROUND_CAP = 3000

final_test = np.zeros(len(te))
for name in best_two:
    params = GRID[name]
    base_rounds = int(np.mean(rounds["2022-03-01", name]))
    n_rounds = min(int(base_rounds * ROUND_SCALE), ROUND_CAP)
    preds = []
    for seed in FINAL_SEEDS:
        model = lgb.LGBMClassifier(**params, n_estimators=n_rounds, random_state=seed)
        model.fit(tr[FEAT], y)
        preds.append(percentile(model.predict_proba(te[FEAT])[:, 1]))
        print(f"{name} seed={seed} rounds={n_rounds} done | {(time.time()-T0)/60:.1f} мин", flush=True)
    final_test += np.mean(preds, axis=0) / len(best_two)
    # контроль на марте с финальным числом итераций (1 seed)
    hold_mask = np.flatnonzero(days == "2022-03-01")
    prior_mask = np.flatnonzero(days < "2022-03-01")
    chk = lgb.LGBMClassifier(**params, n_estimators=n_rounds, random_state=FINAL_SEEDS[0])
    chk.fit(tr[FEAT].iloc[prior_mask], y[prior_mask])
    p = percentile(chk.predict_proba(tr[FEAT].iloc[hold_mask])[:, 1])
    ap, buyers = metrics(y[hold_mask], p, tr[cid].to_numpy()[hold_mask])
    print(f"контроль {name} на марте (n_rounds={n_rounds}): AP@20%={ap:.5f}", flush=True)


def save_submission(filename, scores):
    assert len(scores) == len(sample) and np.isfinite(scores).all()
    assert te[cid].equals(sample[cid])
    pd.DataFrame({cid: te[cid], "score": percentile(scores)}).to_csv(OUT_DIR / filename, index=False)


save_submission("submission.csv", final_test)
print("submission.csv готов.", f"| всего {(time.time()-T0)/60:.1f} мин", flush=True)