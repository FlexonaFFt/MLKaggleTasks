"""Сборка _pipeline_v9_max.ipynb из _pipeline_v8_safe.ipynb + новые ячейки."""
import json
from pathlib import Path

NB_DIR = Path("/Users/flexonafft/MLKaggleTasks/competitions/chempionats/cu-lamoda-intern/notebooks")
src = json.loads((NB_DIR / "_pipeline_v8_safe.ipynb").read_text())
out = {"cells": [], "metadata": src["metadata"], "nbformat": 4, "nbformat_minor": src.get("nbformat_minor", 5)}


def md(s):
    out["cells"].append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})


def code(s):
    out["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)})


# --- ячейки 0..15 v8 (импорты, данные, весь банк признаков, basket-фичи) ---
for c in src["cells"][:16]:
    out["cells"].append(c)

out["cells"][0]["source"] = """# Lamoda · v9 max: сетка гиперпараметров + большой ансамбль

База: полный банк признаков v8 (без basket-колонок — на LB они стабильно ухудшали результат;
самый сильный сигнал этих данных уже в базовом банке). Новое в v9:

- сетка из 5 конфигураций LightGBM, проверка на двух holdout-месяцах,
  выбор по худшему месяцу (защита от подгонки под месяц);
- финал: топ-2 конфиги × 8 seed, обучение на всех трёх месяцах;
- число итераций финала = best_iteration марта × 1.2 (поправка на refit с 90% на 100% данных).

Validation: февраль выбирает, март проверяет.
""".splitlines(keepends=True)

md("""## Валидация: сетка конфигураций LightGBM

Holdout-протокол как в v8: февраль (обучение на январе) и март (обучение на январе+феврале),
ранняя остановка по внутреннему 10% обучающих месяцев, 2 seed, усреднение перцентилей.
Конфиг выбираем по худшему из двух месяцев: если конфиг лучше только на одном месяце —
это подгонка, а не сигнал (локальная AP@20% имеет шум ±0.005–0.01).""")

code('''from sklearn.model_selection import train_test_split
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
print("\\nСредний AP@20% по месяцам (min_month — защита от подгонки):")
display(pivot)
grid_results.to_csv(OUT_DIR / "grid_v9.csv", index=False)
best_two = list(pivot.index[:2])
print("Финал: конфиги", best_two)''')

md("""## Финальный ансамбль

Топ-2 конфиги × 8 seed, обучение на всех трёх месяцах. Число итераций = best_iteration марта × 1.2
(поправка на refit с 90% на 100% данных), потолок 3000. Скор — среднее перцентилей по всем моделям.
Контроль: каждый конфиг дополнительно проверяется на марте с финальным числом итераций.""")

code('''FINAL_SEEDS = tuple(range(42, 50))  # 8 seed
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
print("submission.csv готов.", f"| всего {(time.time()-T0)/60:.1f} мин", flush=True)''')

(NB_DIR / "_pipeline_v9_max.ipynb").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("written", NB_DIR / "_pipeline_v9_max.ipynb", "| cells:", len(out["cells"]))
