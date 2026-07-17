# ROGII Wellbore Geology Prediction — experiment archive

Последнее обновление: 2026-07-16.

## Итог

- Лучший подтверждённый Kaggle public score: **7.043**, submission `54738967`.
- Notebook: `flexonafft/rogii-kim-om020`.
- Предыдущие public scores: `7.099`, `7.110`, `7.113`, `7.148`, `7.201`.
- `7.043` нельзя считать честной оценкой hidden-test: public test содержит 3 wells, все имеют train-копии.
- В notebook `rogii-kim-om020` guarded contact override сработал для всех 3 wells и заменил все `14 151` prediction rows.
- Model package correction был отключён diff-gate: `p95_abs_diff = 26.701 > 25.0`.
- Visible-prefix calibration практически не изменила submission.
- Spatial kriging не запустился: `offset_krige_lookup.csv` не был подключён.

## Что пробовали

### 1. Validation и baseline

`exp_001_prefix_validation` показал очень низкий RMSE `0.005370`, но validation нереалистичен: same-well formation surfaces раскрывают target shape. Результат не использовать для оценки hidden-test.

`exp_002_legal_schema_validation` удалил train-only formation columns и установил legal baseline `last_value = 11.493684`.

### 2. GR alignment и beam search

`exp_003_gr_alignment` сравнил PF, beam и безопасные blend-варианты на multi-cut validation. Лучший вариант `last_beam_w25`: pooled RMSE `16.065684`; standalone PF отвергнут.

`exp_005_7148_bounded_beam_v1` улучшил public score `7.148` до `7.110`.

`exp_007_beam_policy_compare` не дал улучшения: лучший pooled RMSE `20.815831`.

### 3. Selector и gating

- Binary gate (`exp_008`): pooled RMSE `20.485260`; лучше continuous weight, но недостаточно для уверенного public submission.
- Continuous weight (`exp_009`): `20.542858`.
- Veto selector (`exp_010`): `20.814866`; отклонён.

### 4. Public anchor и model package

`exp_012_gold_7039_clean` воспроизвёл reported notebook score `7.039`, но фактическая Kaggle submission получила `7.099`.

`exp_014_model_package_oof` показал signal у model package, но тяжёлый failure tail: pooled RMSE `10.670211`.

`exp_015_anchor_package_oof` выровнял anchor и package OOF; package признан недостаточно стабильным для blind blend.

`exp_016_ridge_stage_package` отправлен на Kaggle; public score `7.113`.

### 5. Лучший Kaggle notebook

`flexonafft/rogii-kim-om020`, submission `54738967`:

- profile: `vp_balanced_modelpkg_010`;
- 128-seed likelihood-weighted PF;
- Ridge/PF anchor;
- learned trajectory branch;
- projection;
- guarded contact override;
- visible-prefix calibration;
- model-package branch с защитным diff-gate.

Фактически score `7.043` в основном объясняется contact override на public same-well overlap. Это public anchor, не доказанное hidden-test улучшение.

## Текущая стратегия

1. Сохранить `7.043` как public anchor.
2. Не тюнить по одному public score.
3. Построить честный GroupKFold по well с несколькими synthetic prediction starts.
4. Сравнивать baseline, beam, PF, Ridge, learned branch и contact-free варианты на одинаковых cuts.
5. Сохранять OOF predictions, worst-decile RMSE и error correlations.
6. Selector обучать только cross-fitted OOF.
7. Contact override использовать отдельно как known-overlap branch, не смешивать с hidden-test оценкой.

## exp_017_clean_oof — Kaggle result

Kernel `flexonafft/rogii-clean-oof-benchmark-017` завершился успешно. В clean-режиме:

- contact override: `False`;
- visible-prefix calibration: `False`;
- model package correction: `False`;
- kriging: отсутствует.

На 250 train wells и `1 198 217` suffix rows:

- selector CV pooled RMSE: **9.310996**;
- per-well RMSE p50: `5.725292`;
- per-well RMSE p90: `13.264390`;
- per-well RMSE p99: `29.149480`;
- worst-decile SSE share: `53.87495%`.

Oracle diagnostics дают line `6.731998` и smooth `3.034980`, но используют известные suffix targets и являются верхней границей, не deployable score.

Вывод: core stack работает без public overlap, но до public `7.043` не дотягивает. Следующий полезный шаг — анализ worst-decile wells и selector features, не новый blind Kaggle submission.

## exp_018_hardwell_gate — Kaggle result

Kernel завершился успешно. Исправлен скрытый leakage в `exp_017`: same-well physical candidate теперь обнуляется до формирования prediction rows.

- contact override: `False`;
- visible-prefix calibration: `False`;
- model package correction: `False`;
- hard-well gate: `True`;
- gated wells: `1 / 3` public wells;
- anchor fallback weight: `0.65`;
- public score: не отправлялся.

Gate report:

- `000d7d20`: not gated;
- `00bbac68`: gated;
- `00e12e8b`: not gated.

Это target-free smoke test, не доказательство улучшения: public test содержит только 3 wells, а gate threshold взят как test quantile `0.85`. Нужна cross-fitted OOF calibration на train wells перед blind submission.

## exp_019_hardwell_gate_oof — Kaggle result

OOF calibration завершилась успешно и gate был выбран:

- выбранный quantile: `0.70`;
- anchor weight: `0.25`;
- gated OOF wells: `232`;
- pooled RMSE: `10.372253 -> 9.845537`;
- worst-decile SSE share: `0.569541 -> 0.564714`;
- RMSE gain: `0.526716`;
- tail gain: `0.004827`.

На public test gate применился к `1 / 3` wells. Submission audit прошёл, public score не отправлялся.

Оговорка: OOF calibration fallback использует `last_known_tvt`, а test gate смешивает с projected SP45 anchor. Поэтому результат подтверждает направление uncertainty gate, но не является полностью aligned OOF оценкой. Следующий эксперимент должен калибровать именно тот же SP45 fallback.

## exp_020_hardwell_gate_aligned — Kaggle result

Выравнивание fallback с SP45 улучшило результат:

- fallback: `aligned_sp45_proxy`;
- quantile: `0.80`;
- anchor weight: `0.35`;
- pooled RMSE: `10.372253 -> 9.761387`;
- RMSE gain: `0.610866`;
- worst-decile SSE share: `0.569541 -> 0.563247`;
- tail gain: `0.006294`.

На public test gate применился к `1 / 3` wells. Submission audit прошёл. Public score ещё не отправлялся.

Это первый gate-вариант, который можно отправить как controlled hidden-robust submission. Остаются две оговорки: fallback — proxy SP45, не полный train OOF selector; threshold/weight выбирались на том же OOF, без nested holdout.

## Public submission 54767301 — score 7.679

`exp_020` был отправлен без same-well contact override и получил public RMSE `7.679`. Это на `0.636` хуже `7.043` из submission `54738967`.

Причина ожидаема: public test состоит из 3 wells, все имеют train-копии. `7.043` заменял все public prediction rows contact path; `exp_020` этот источник отключил. OOF gain `10.372253 -> 9.761387` не переносится на public overlap distribution.

Public leaderboard и hidden-robust validation нужно вести как две разные ветки. Для public score восстановить contact-gated anchor; gate разрешать только после contact override или на wells без overlap.

## public_anchor_v2 — Kaggle output

Notebook завершился корректно, но output оказался byte-identical с `7.043` anchor:

- все 3 wells выбрали `EGFDU`;
- contact override: `3 / 3`, все `14 151` rows;
- submission SHA-256: `fdf4a8175b6ec6a70c9b78fd6916ac3c317e43f7e9c08bbca87cd02314801ca9`;
- spatial kriging skipped: lookup не подключён.

Новый formation selector не дал изменения. Повторно отправлять этот submission не нужно.

## Сохранено вне этого архива

`datasets/` не удалён. Он содержит локальный Kaggle dataset и нужен для дальнейшего обучения. Остальные старые experiment artifacts удалены после создания этого summary.

## Research hypotheses v1

Diagnostic Kaggle kernel: [rogii-research-hypotheses-v1](https://www.kaggle.com/code/flexonafft/rogii-research-hypotheses-v1).

OOF pseudo-test results on 773 train wells:

- `recent_slope`: `84.415559` — reject;
- `last_known`: `12.873392` — safer baseline;
- `model_direction`: `12.491620` — small improvement;
- `blend_guarded`: `12.425053` — best screened variant;
- `neighbor_copy`: `18.319650` — reject in current geometry/transfer form;
- `midpoint_gate`: `13.003186` — reject as currently implemented.

This is not a Kaggle score. The notebook is a research screen and creates no submission. Exact formation-contact reconstruction and CNN/particle-filter were not tested because required artifacts are not mounted.

## Research hypotheses v2 - faithful GR alignment

Kaggle kernel: [rogii-research-hypotheses-v2-gr-alignment](https://www.kaggle.com/code/flexonafft/rogii-research-hypotheses-v2-gr-alignment).

Implemented the core GR beam hypothesis with two candidate TVT paths, near-tie midpoint, and aligned neighbor transfer. Kaggle completed all 5 OOF folds and exact public forensic scoring.

OOF results:

- `last_known`: `15.596899`;
- `beam_1`: `15.688177`;
- `beam_midpoint`: `15.688251`;
- `neighbor_guarded`: `16.725163`.

Exact public train-copy forensic results (weighted pooled RMSE): beam `11.505831`; midpoint `11.505835`. This is materially worse than the known `7.043` contact anchor. The near-tie detector fired on `100%` of sampled rows, so its current cost-gap calibration is not informative. Do not submit this branch.

## Contact forensics v3

Kaggle kernel: [rogii-contact-forensics-v3](https://www.kaggle.com/code/flexonafft/rogii-contact-forensics-v3).

This reproduced the original contact equation and screened all formations with prefix mean/median/trimmed/linear offsets plus an oracle full-train offset. Kaggle completed successfully.

The train suffix OOF contact equation is extremely strong (`~0.006 ft`), and the exact train-copy forensic is also `~0.005 ft`. However, this does not match the real Kaggle public score `7.043`; therefore train `TVT` in the copied wells is not a valid proxy for hidden public suffix labels. Prefix-selected surfaces were `EGFDL`, `ASTNL`, `EGFDL`; oracle-selected surfaces differed again. No new submission is justified. Keep `7.043` anchor.
