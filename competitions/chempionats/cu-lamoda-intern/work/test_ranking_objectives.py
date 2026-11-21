"""Binary vs lambdarank vs rank_xendcg на одних и тех же 36 фичах.
Validation: train Jan(+Feb) -> holdout Mar (и Feb).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import reference_rf as R  # пересоздаст фичи и напечатает свой результат; игнорируем

# reference_rf уже построил tr/te и cols при импорте
tr, te, cols = R.tr, R.te, R.cols
ap20 = R.ap20

params_common = dict(learning_rate=.05, num_leaves=47, min_child_samples=85,
                     colsample_bytree=.6, subsample=.89, subsample_freq=1, reg_lambda=.66,
                     min_split_gain=.43, n_jobs=8, verbose=-1)

for hold_month in ["2022-02-01", "2022-03-01"]:
    prior = tr[tr.scoring_date < hold_month]
    hold = tr[tr.scoring_date == hold_month].reset_index(drop=True)
    print(f"\n=== holdout {hold_month} ===")

    # 1) binary baseline
    dtrain = lgb.Dataset(prior[cols], prior.target)
    m = lgb.train({**params_common, "objective": "binary"}, dtrain, num_boost_round=2000,
                  valid_sets=[lgb.Dataset(hold[cols], hold.target)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict(hold[cols], num_iteration=m.best_iteration)
    print(f"binary:      iters={m.best_iteration:4d} AP@20%={ap20(hold.target.to_numpy(), p, hold.customer_id.to_numpy()):.5f}")

    # 2) lambdarank: группы по 5000 строк (лимит 10000/группу), label = target
    def grouped(df):
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        gs, cur = [], df
        while len(cur):
            gs.append(min(5000, len(cur)))
            cur = cur.iloc[gs[-1]:]
        return df, gs

    for obj in ["lambdarank", "rank_xendcg"]:
        pr, gtrain = grouped(prior)
        ho, gvalid = grouped(hold)
        dtrain_r = lgb.Dataset(pr[cols], pr.target, group=gtrain)
        dvalid_r = lgb.Dataset(ho[cols], ho.target, group=gvalid, reference=dtrain_r)
        mp = {**params_common, "objective": obj, "eval_at": [20], "label_gain": [0, 1]}
        try:
            m2 = lgb.train(mp, dtrain_r, num_boost_round=2000, valid_sets=[dvalid_r],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            p2 = m2.predict(hold[cols], num_iteration=m2.best_iteration)
            print(f"{obj}: iters={m2.best_iteration:4d} AP@20%={ap20(hold.target.to_numpy(), p2, hold.customer_id.to_numpy()):.5f}")
        except Exception as e:
            print(f"{obj}: FAILED {e}")
