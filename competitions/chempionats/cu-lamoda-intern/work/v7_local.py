'''
Пайплайн v7 = v5/v6 + прорывные фичи по ритму покупок.

База: v5 (485 признаков), CV смесь+правило 0.7603 / Public 0.74961.
Новое относительно v6:
1) автопсия генератора таргета для сегмента 4+ (таблицы precision);
2) multi-cycle renewal-фичи (жёсткие и мягкие вероятности попадания в окно);
3) правило CORE4 для 4+, если OOF показывает плюс;
4) быстрый гейт: 1 seed LightGBM на holdout Feb и Mar — смотреть до полного ансамбля.

Запуск целиком на Kaggle T4: ~55–70 минут. Первая строка вывода: GPU: True.
Для проверяющего: весь путь от данных до submission.csv в этом ноутбуке.

Шаги:
1. признаки v5;
2. автопсия 4+ + renewal/CORE4 фичи;
3. быстрый гейт Feb/Mar;
4. CV: LightGBM, LightGBM по сегментам, CatBoost (5 фолдов × 5 зёрен);
5. смесь по рангам + правила 2–3 и 4+; submission.csv и запасные файлы.
'''
import itertools
import json
import os
import re
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
def display(*a, **k): print(*[x.to_string() if hasattr(x,'to_string') else str(x) for x in a] if a else '')
from scipy.special import ndtr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

KAGGLE_DIR = Path("/kaggle/input/competitions/lamoda-cu-repeat-purchase-2026")
_LOCAL = Path("data") if Path("data/train.csv").exists() else Path("../data")
DATA_DIR = KAGGLE_DIR if KAGGLE_DIR.exists() else Path(os.environ.get("LAMODA_DATA_DIR", _LOCAL))
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(os.environ.get("LAMODA_OUT_DIR", "."))
print("DATA_DIR:", DATA_DIR)

SEED = 42
GAP_DAYS = 14
TARGET_MONTHS = 3
WINDOWS = [7, 14, 30, 60, 90, 180, 365]
N_FOLDS = 5
N_SEEDS = 5                                    # зёрен на модель
EARLY_STOP = 200
try:
    GPU = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
except Exception:
    GPU = False
print("GPU:", GPU)
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

# Новый признак остаётся, если его важность выше 90-го процентиля важности на перемешанном таргете
trd = data[data["part"].eq("train")]
Xs, ys = trd[BASE + list(NAMES)], trd["target"].astype(int).to_numpy()
sel = dict(objective="binary", learning_rate=0.05, n_estimators=300, num_leaves=31, min_child_samples=100,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=-1, verbose=-1)


def gain(target, seed):
    m = lgb.LGBMClassifier(**sel, random_state=seed).fit(Xs, target)
    return pd.Series(m.booster_.feature_importance("gain"), index=Xs.columns)


rng = np.random.default_rng(SEED)
real = pd.concat([gain(ys, s) for s in range(3)], axis=1).mean(axis=1)
null = pd.concat([gain(rng.permutation(ys), s) for s in range(6)], axis=1)
ratio = real / (null.quantile(0.9, axis=1) + 1e-9)
KEEP_NEW = [c for c in NAMES if ratio[c] > 1.0]
FEATURES = BASE + KEEP_NEW
print(f"новых признаков выше шума: {len(KEEP_NEW)} из {len(NAMES)} | всего признаков: {len(FEATURES)} | {(time.time() - T0) / 60:.1f} мин")
print("Самые важные из новых:")
display((real[KEEP_NEW] / real.sum()).sort_values(ascending=False).head(20).rename(index=NAMES).to_frame("доля важности"))

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

# Автопсия: для train считаем hit-rate target по решётке period × m × overdue
INV_ALL = {v: k for k, v in NAMES.items()}  # короткие имена unit_feats
periods = ["mean", "median", "max", "trim", "q75", "last", "ewm"]
hist = "год"
rows = []
train_mask = data["part"].eq("train")
sub = data.loc[train_mask].copy()
lo = float(GAP_DAYS)
hi = (GAP_DAYS + sub["win_len"]).astype(float)
r = sub["recency"].astype(float)
n = sub["n_orders"].astype(int)
y_ = sub["target"].astype(int)

for seg_name, seg_m in [("4-8", (n >= 4) & (n <= 8)), ("9+", n >= 9), ("4+", n >= 4)]:
    for p in periods:
        key = f"заказ|{hist}|{p}|t"
        if key not in INV_ALL:
            continue
        # t = period - recency  =>  period = t + recency, next at recency + period = 2*period - t? 
        # В unit_feats: tn = P - r, так что ожидаемый день после cutoff = tn.
        # Цикл m: следующая покупка на tn + (m-1)*P = m*P - r
        tcol = INV_ALL[key]
        tn = sub[tcol].astype(float)
        P = tn + r  # period
        overdue = r > P
        for m in (1, 2, 3):
            hit = (m * P - r)
            inn = (hit > lo) & (hit <= hi)
            for branch, bm in [("all", pd.Series(True, index=sub.index)),
                               ("in_window", inn),
                               ("overdue", overdue),
                               ("not_overdue", ~overdue),
                               ("in_and_overdue", inn & overdue),
                               ("in_and_fresh", inn & ~overdue)]:
                sel = seg_m & bm & P.notna() & (P > 0)
                if sel.sum() < 80:
                    continue
                prec = float(y_[sel].mean())
                rows.append({
                    "seg": seg_name, "period": p, "m": m, "branch": branch,
                    "n": int(sel.sum()), "precision": prec,
                    "lift": prec / max(float(y_[seg_m].mean()), 1e-9),
                })

audit = pd.DataFrame(rows).sort_values(["precision", "n"], ascending=[False, False])
print("Топ правил по precision (n>=80):")
display(audit.head(40))
# лучшие почти-детерминированные
good = audit[(audit["precision"] >= 0.85) & (audit["n"] >= 100)].head(20)
print("Кандидаты CORE4 (precision>=0.85, n>=100):")
display(good)
audit.to_csv(OUT_DIR / "autopsy_4plus.csv", index=False)

# Зафиксируем лучшее правило для CORE4: среди seg=4+, branch с max precision*log(n)
cand = audit[audit["seg"].eq("4+") & audit["branch"].isin(["in_window", "in_and_overdue", "in_and_fresh", "overdue"])].copy()
cand["score"] = cand["precision"] * np.log1p(cand["n"])
BEST_RULE = cand.sort_values("score", ascending=False).iloc[0].to_dict() if len(cand) else {
    "period": "max", "m": 1, "branch": "in_window", "precision": 0.0, "n": 0
}
print("Выбрано для CORE4:", BEST_RULE)



# ====== CELL BREAK ======

# --- multi-cycle renewal features (компактно: ~30 штук) ---
RENEW_COLS = []
r_all = data["recency"].astype(float)
lo_all = float(GAP_DAYS)
hi_all = (GAP_DAYS + data["win_len"]).astype(float)


def soft_hit(center, lo, hi, sigma):
    sig = sigma.replace(0, np.nan)
    return (ndtr((hi - center) / sig) - ndtr((lo - center) / sig)).clip(0, 1)


for hist in ["вся", "год"]:
    for p in ["mean", "median", "max", "trim", "ewm"]:
        t_key = f"заказ|{hist}|{p}|t"
        if t_key not in INV_ALL:
            continue
        tn = data[INV_ALL[t_key]].astype(float)
        P = (tn + r_all).where(lambda s: s > 0)
        hard_any = pd.Series(0.0, index=data.index)
        e_count = pd.Series(0.0, index=data.index)
        soft_prod = pd.Series(1.0, index=data.index)
        soft_sum = pd.Series(0.0, index=data.index)
        for m in (1, 2, 3):
            center = m * P - r_all
            hard = ((center > lo_all) & (center <= hi_all)).astype(float).where(P.notna())
            hard_any = np.maximum(hard_any, hard.fillna(0))
            e_count = e_count + hard.fillna(0)
            sh = soft_hit(center, lo_all, hi_all, 0.25 * P)
            soft_sum = soft_sum + sh.fillna(0)
            soft_prod = soft_prod * (1.0 - sh.fillna(0))
        # ближайший центр среди m=1..3
        c1, c2, c3 = P - r_all, 2 * P - r_all, 3 * P - r_all
        dist1 = (lo_all - c1).clip(lower=0) + (c1 - hi_all).clip(lower=0)
        dist2 = (lo_all - c2).clip(lower=0) + (c2 - hi_all).clip(lower=0)
        dist3 = (lo_all - c3).clip(lower=0) + (c3 - hi_all).clip(lower=0)
        min_dist = np.fmin(np.fmin(dist1, dist2), dist3)

        data[f"rnw|{hist}|{p}|any_hard"] = hard_any.astype("float32")
        data[f"rnw|{hist}|{p}|n_hard"] = e_count.astype("float32")
        data[f"rnw|{hist}|{p}|p_any_soft"] = (1.0 - soft_prod).clip(0, 1).astype("float32")
        data[f"rnw|{hist}|{p}|e_soft"] = soft_sum.astype("float32")
        data[f"rnw|{hist}|{p}|min_dist"] = min_dist.astype("float32")
        data[f"rnw|{hist}|{p}|overdue"] = (r_all > P).astype("float32")
        data[f"rnw|{hist}|{p}|rec_over_P"] = (r_all / P).astype("float32")
        RENEW_COLS += [
            f"rnw|{hist}|{p}|any_hard", f"rnw|{hist}|{p}|n_hard",
            f"rnw|{hist}|{p}|p_any_soft", f"rnw|{hist}|{p}|e_soft",
            f"rnw|{hist}|{p}|min_dist", f"rnw|{hist}|{p}|overdue",
            f"rnw|{hist}|{p}|rec_over_P",
        ]

RENEW_SHORT = {f"r{i}": c for i, c in enumerate(RENEW_COLS)}
data = data.rename(columns={v: k for k, v in RENEW_SHORT.items()})
RENEW_FEATURES = list(RENEW_SHORT)
NAMES.update(RENEW_SHORT)
INV_ALL = {v: k for k, v in NAMES.items()}

# --- CORE4 from BEST_RULE ---
bp = BEST_RULE.get("period", "max")
bm = int(BEST_RULE.get("m", 1))
bb = BEST_RULE.get("branch", "in_window")
t_key = f"заказ|год|{bp}|t"
if t_key not in INV_ALL:
    t_key = "заказ|вся|max|t"
tn = data[INV_ALL[t_key]].astype(float)
P4 = tn + data["recency"].astype(float)
hit4 = bm * P4 - data["recency"].astype(float)
inn4 = (hit4 > lo_all) & (hit4 <= hi_all)
od4 = data["recency"].astype(float) > P4
if bb == "overdue":
    core4_mask = od4
elif bb == "in_and_overdue":
    core4_mask = inn4 & od4
elif bb == "in_and_fresh":
    core4_mask = inn4 & ~od4
else:
    core4_mask = inn4
core4_mask = core4_mask & (data["n_orders"] >= 4) & P4.notna() & (P4 > 0)

data["core4_flag"] = core4_mask.astype("float32")
data["core4_t"] = hit4.astype("float32")
data["core4_dist"] = ((lo_all - hit4).clip(lower=0) + (hit4 - hi_all).clip(lower=0)).astype("float32")
data["core4_overdue"] = od4.astype("float32")
CORE4_FEATURES = ["core4_flag", "core4_t", "core4_dist", "core4_overdue"]

FEATURES = list(dict.fromkeys(FEATURES + RENEW_FEATURES + CORE4_FEATURES))
print(f"renewal: {len(RENEW_FEATURES)} | core4 rule={BEST_RULE} | признаков: {len(FEATURES)} | {(time.time()-T0)/60:.1f} мин")
print("core4 share train/test:",
      float(data.loc[data.part.eq("train"), "core4_flag"].mean()),
      float(data.loc[data.part.eq("test"), "core4_flag"].mean()))



# ====== CELL BREAK ======

# Быстрый гейт: сравнение FEATURES vs FEATURES без renewal/core4 на временных holdout
FEAT_BASE = [c for c in FEATURES if c not in RENEW_FEATURES + CORE4_FEATURES]
FEAT_NEW = FEATURES

tr_all = data[data["part"].eq("train")].reset_index(drop=True)


def holdout_ap(feature_list, train_slices, valid_slice, seed=42):
    trn = tr_all[tr_all["slice"].isin(train_slices)]
    val = tr_all[tr_all["slice"] == valid_slice]
    Xa, ya = trn[feature_list], trn["target"].astype(int)
    Xb, yb = val[feature_list], val["target"].astype(int)
    # категории как в lgb-пути: top_category2 уже category
    model = lgb.LGBMClassifier(**{**PARAMS_GATE, "random_state": seed})
    model.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = model.predict_proba(Xb)[:, 1]
    return fast_ap(yb.to_numpy(), pred), float(roc_auc_score(yb, pred))


PARAMS_GATE = dict(objective="binary", learning_rate=0.05, n_estimators=2000, subsample_freq=1,
                   num_leaves=47, min_child_samples=85, colsample_bytree=0.8, subsample=0.89,
                   reg_lambda=0.66, reg_alpha=0.0046, n_jobs=-1, verbose=-1)

gate_rows = []
for tag, feats in [("base_v5", FEAT_BASE), ("v7_renewal", FEAT_NEW)]:
    for valid, trains in [("2022-02-01", ["2022-01-01"]), ("2022-03-01", ["2022-01-01", "2022-02-01"])]:
        ap, auc = holdout_ap(feats, trains, valid)
        gate_rows.append({"set": tag, "valid": valid, "AP@20%": ap, "AUC": auc})
        print(f"{tag} | {valid}: AP={ap:.4f} AUC={auc:.4f}", flush=True)

gate = pd.DataFrame(gate_rows)
display(gate.pivot(index="valid", columns="set", values="AP@20%"))
delta = gate.pivot(index="valid", columns="set", values="AP@20%")
lift = (delta["v7_renewal"] - delta["base_v5"]) if "v7_renewal" in delta and "base_v5" in delta else None
print("Δ AP@20% (v7 - base):")
display(lift)
gate.to_csv(OUT_DIR / "gate_v7.csv", index=False)
USE_NEW = True
if lift is not None:
    USE_NEW = bool((lift >= -0.002).all() and (lift.max() >= 0.001))
print("Берём renewal+core4 в полный прогон:", USE_NEW)
if not USE_NEW:
    FEATURES = FEAT_BASE
    print("Откат к base_v5, признаков:", len(FEATURES))
else:
    print("Идём с v7, признаков:", len(FEATURES))



# ====== CELL BREAK ======

tr = data[data["part"].eq("train")].reset_index(drop=True)
te = data[data["part"].eq("test")].set_index("customer_id").loc[sample["customer_id"]].reset_index()
y = tr["target"].astype(int).to_numpy()
sl = tr["slice"].to_numpy()


def frame(kind, df):
    # CatBoost получает ещё бренд и берёт категориальные признаки строками; остальным хватает подкатегории
    if kind != "cb":
        return df[FEATURES]
    out = df[FEATURES + ["top_brand"]].copy()
    out["top_category2"] = out["top_category2"].astype(str)
    return out


X = {k: frame(k, tr) for k in ["lgb", "xgb", "cb"]}
X_TE = {k: frame(k, te) for k in ["lgb", "xgb", "cb"]}


def segment(df):
    # 0 — один заказ, 1 — два-три, 2 — четыре и больше
    return np.select([df["n_orders"] == 1, df["n_orders"] <= 3], [0, 1], 2)


def core(df):
    # клиенты с 2–3 заказами, у которых следующий заказ по средней паузе приходится на окно
    t = df[INV["заказ|вся|mean|t"]]
    return (df["n_orders"].between(2, 3) & (t > 35) & (t <= 85)).to_numpy()


def core4(df):
    # буст для 4+ по флагу из автопсии (если фича есть)
    if "core4_flag" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df["core4_flag"].astype(bool).to_numpy()


INV = {v: k for k, v in NAMES.items()}
SEG, SEG_TE = segment(tr), segment(te)
CORE, CORE_TE = core(tr), core(te)
CORE4, CORE4_TE = core4(tr), core4(te)
print("признаков:", len(FEATURES), "| train:", len(tr), "| test:", len(te),
      f"| правило 2–3: train {CORE.mean():.2%}, test {CORE_TE.mean():.2%}",
      f"| CORE4: train {CORE4.mean():.2%}, test {CORE4_TE.mean():.2%}",
      f"| {(time.time() - T0) / 60:.1f} мин")


def score(pred):
    # среднее AP@20% по срезам
    return float(np.mean([fast_ap(y[sl == s], pred[sl == s]) for s in np.unique(sl)]))

# ====== CELL BREAK ======

PARAMS = {
    "lgb": dict(objective="binary", learning_rate=0.02, n_estimators=8000, subsample_freq=1, cat_smooth=50, n_jobs=-1, verbose=-1,
                num_leaves=47, min_child_samples=85, colsample_bytree=0.8, subsample=0.89, reg_lambda=0.66, reg_alpha=0.0046,
                min_split_gain=0.43, max_bin=255),
    "xgb": dict(learning_rate=0.03, n_estimators=5000, tree_method="hist", device="cuda" if GPU else "cpu", enable_categorical=True,
                eval_metric="logloss", early_stopping_rounds=EARLY_STOP, n_jobs=-1, verbosity=0,
                max_depth=7, min_child_weight=23.76, subsample=0.853, colsample_bytree=0.828, reg_lambda=0.177, reg_alpha=2.89,
                gamma=0.572, max_bin=256),
    "cb": dict(loss_function="Logloss", learning_rate=0.05, iterations=5000, bootstrap_type="Bayesian", task_type="GPU" if GPU else "CPU",
               cat_features=["top_category2", "top_brand"], thread_count=-1, verbose=0,
               depth=7, l2_leaf_reg=1.37, random_strength=0.134, bagging_temperature=0.0174, border_count=64),
}


def fit(kind, a, b, seed):
    # обучение на строках a с ранней остановкой по строкам b; возвращает функцию прогноза
    key = "lgb" if kind == "seg" else kind
    Xa, Xb = X[key].iloc[a], X[key].iloc[b]
    if key == "lgb":
        m = lgb.LGBMClassifier(**PARAMS["lgb"], random_state=seed).fit(Xa, y[a], eval_set=[(Xb, y[b])],
                                                                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    elif key == "xgb":
        m = xgb.XGBClassifier(**PARAMS["xgb"], random_state=seed).fit(Xa, y[a], eval_set=[(Xb, y[b])], verbose=False)
    else:
        m = CatBoostClassifier(**PARAMS["cb"], random_seed=seed).fit(Xa, y[a], eval_set=(Xb, y[b]), early_stopping_rounds=EARLY_STOP)
    return lambda Z: m.predict_proba(Z)[:, 1]


strata = tr["slice"] + "_" + tr["target"].astype(int).astype(str)


def cv_run(kind):
    # прогнозы вне фолда на train и прогноз на тест, усреднённые по зёрнам; "seg" — отдельная LightGBM на каждый сегмент
    key = "lgb" if kind == "seg" else kind
    oof, test = np.zeros(len(tr)), np.zeros(len(te))
    for seed in range(N_SEEDS):
        for a, b in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + seed).split(tr, strata):
            groups = [(a, b, np.ones(len(te), bool))] if kind != "seg" else [(a[SEG[a] == s], b[SEG[b] == s], SEG_TE == s) for s in range(3)]
            for aa, bb, tt in groups:
                predict = fit(kind, aa, bb, SEED + seed)
                oof[bb] += predict(X[key].iloc[bb]) / N_SEEDS
                test[tt] += predict(X_TE[key][tt]) / (N_SEEDS * N_FOLDS)
    return oof, test

# ====== CELL BREAK ======

RES, rows = {}, []
for kind in ["lgb", "seg", "cb"]:
    t0 = time.time()
    oof, test = cv_run(kind)
    RES[kind] = dict(oof=oof, test=test)
    rows.append({"модель": kind, "AP@20%": score(oof), **{s: fast_ap(y[sl == s], oof[sl == s]) for s in np.unique(sl)},
                 "минут": round((time.time() - t0) / 60, 1)})
    print(f"{kind}: {score(oof):.4f} | {(time.time() - t0) / 60:.1f} мин", flush=True)
print(f"AP@20% на кросс-валидации, {N_FOLDS} фолдов, зёрен: {N_SEEDS}:")
display(pd.DataFrame(rows).set_index("модель"))

# ====== CELL BREAK ======

names = list(RES)
rank_oof = np.column_stack([pd.Series(RES[k]["oof"]).groupby(sl).rank(pct=True).to_numpy() for k in names])
rank_te = np.column_stack([pd.Series(RES[k]["test"]).rank(pct=True).to_numpy() for k in names])
equal = np.full(len(names), 1 / len(names))
grid = [np.array(w) / 10 for w in itertools.product(range(11), repeat=len(names)) if sum(w) == 10]
w_best = max(grid, key=lambda w: score(rank_oof @ w))
blend_oof, blend_te = rank_oof @ w_best, rank_te @ w_best
rule23_oof, rule23_te = blend_oof + CORE, blend_te + CORE_TE
rule4_oof, rule4_te = blend_oof + CORE4, blend_te + CORE4_TE
rule_both_oof, rule_both_te = blend_oof + CORE + CORE4, blend_te + CORE_TE + CORE4_TE

variants = {
    **{k: rank_oof[:, i] for i, k in enumerate(names)},
    "равная смесь": rank_oof @ equal,
    "смесь с весами": blend_oof,
    "смесь + правило 2–3": rule23_oof,
    "смесь + CORE4": rule4_oof,
    "смесь + оба правила": rule_both_oof,
}
display(pd.DataFrame({k: {"AP@20%": score(v), **{s: fast_ap(y[sl == s], v[sl == s]) for s in np.unique(sl)}} for k, v in variants.items()}).T)
print("веса:", dict(zip(names, np.round(w_best, 2).tolist())))

cand_rules = {
    "none": (blend_oof, blend_te),
    "rule23": (rule23_oof, rule23_te),
    "core4": (rule4_oof, rule4_te),
    "both": (rule_both_oof, rule_both_te),
}
best_rule_name = max(cand_rules, key=lambda k: score(cand_rules[k][0]))
use_rule = best_rule_name != "none"
final_oof, final_te = cand_rules[best_rule_name]
print("выбрано правило:", best_rule_name, "| AP@20%:", f"{score(final_oof):.4f}")
print("Корреляция прогнозов моделей на тесте (Спирмен):")
display(pd.DataFrame(rank_te, columns=names).corr(method="spearman"))


def save(scores, fname):
    sub = pd.DataFrame({"customer_id": te["customer_id"].to_numpy(), "score": pd.Series(scores).rank(pct=True).to_numpy()})
    assert sub["customer_id"].tolist() == sample["customer_id"].tolist() and sub["score"].between(0, 1).all()
    sub.to_csv(OUT_DIR / fname, index=False)


save(rule_te if use_rule else blend_te, "submission.csv")
save(blend_te, "submission_norule.csv")
save(rank_te @ equal, "submission_equal.csv")
for i, k in enumerate(names):
    save(rank_te[:, i], f"submission_{k}.csv")
pd.DataFrame({"customer_id": tr["customer_id"], "slice": sl, "target": y, **{k: RES[k]["oof"] for k in names}}).to_parquet(OUT_DIR / "oof_v6.parquet", index=False)
pd.DataFrame({"customer_id": te["customer_id"], **{k: RES[k]["test"] for k in names}}).to_parquet(OUT_DIR / "test_pred_v6.parquet", index=False)
json.dump({"weights": dict(zip(names, map(float, w_best))), "rule": bool(use_rule), "features": FEATURES}, open(OUT_DIR / "params_v6.json", "w"), indent=1)
print(f"Готово: submission.csv сохранён. Всего времени: {(time.time() - T0) / 60:.1f} мин")

# ====== CELL BREAK ======

best = final_oof
res = pd.DataFrame({"slice": sl, "target": y, "n_orders": tr["n_orders"].to_numpy(), "score": best})
res["pct"] = res.groupby("slice")["score"].rank(pct=True, ascending=False)
res["n_bin"] = pd.cut(res["n_orders"], [0, 1, 2, 3, 5, 9, 20, 10_000], labels=["1", "2", "3", "4-5", "6-9", "10-20", "21+"])
res["in_top"] = res["pct"] <= 0.20
print("Доля target = 1 по положению в рейтинге:")
display(res.groupby(pd.cut(res["pct"], [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]), observed=True)["target"].agg(["mean", "size"]).T)
print("Верхние 20% по группам числа заказов:")
display(pd.DataFrame({
    "клиентов": res.groupby("n_bin", observed=True).size(),
    "доля в топ-20%": res.groupby("n_bin", observed=True)["in_top"].mean(),
    "target в топе": res[res["in_top"]].groupby("n_bin", observed=True)["target"].mean(),
    "target вне топа": res[~res["in_top"]].groupby("n_bin", observed=True)["target"].mean(),
    "AUC внутри группы": res.groupby("n_bin", observed=True).apply(lambda g: roc_auc_score(g["target"], g["score"]) if g["target"].nunique() == 2 else np.nan),
}))