# ROGII Wellbore Geology Prediction — experiment archive

Последнее обновление: 2026-07-30.

## Важная поправка после проверки Kaggle — 2026-07-30

Организатор уточнил, что видимая папка `test/` содержит только примеры для
создания submission; при запуске на Kaggle она заменяется скрытым test set.
Следовательно, same-ID train/test совпадения из локально видимых файлов не
характеризуют оцениваемый public или private набор. Все нижеупомянутые выводы,
которые связывают same-ID overlap с leaderboard, следует читать только как
forensic-анализ authoring-примеров, а не как объяснение Kaggle score. Новые
решения должны исключать такой путь и валидироваться target-free по wells.

Источник: [официальное разъяснение](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/729837#3503960).

## Итог

- Лучший подтверждённый Kaggle public score: **6.979**, submission `54824983` от 2026-07-19.
- Предыдущие ближайшие public scores: `7.003`, `7.010`, `7.043`, `7.099`, `7.110`, `7.113`.
- Submission `54853579` от 2026-07-20 остаётся в статусе `PENDING` и не считается подтверждённым результатом.
- `6.979` нельзя считать честной оценкой unseen-well: public test содержит 3 wells, все имеют same-ID train-копии.
- Сохранённый anchor-файл имеет SHA-256 `0c60510dc11f7750c493c29c75ac9383eb6ea331d976a0c7991a895c700e7cf8`.
- Public leaderboard и исследовательский OOF отвечают на разные вопросы; их значения нельзя сравнивать напрямую.
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

1. Заморозить `6.979` как public-overlap anchor и не смешивать его с unseen-well моделями без отдельного A/B обоснования.
2. Не тюнить hidden-robust модель по трём public wells.
3. Считать `13.351` лучшим результатом random-well neighbor OOF, но помнить, что эта проверка допускает близкие train references.
4. Для заявлений о переносе на новый район использовать outer spatial holdout; его шкала существенно жёстче random-well OOF.
5. Не развивать GR-DTW как основной decoder: использовать GR только как слабый признак режима или confidence.
6. Следующую гипотезу проверять внутри локальных пространственно-формационных режимов: сначала recoverability коэффициентов, затем модель.
7. Любой selector и gate обучать cross-fitted; oracle-результаты использовать только как потолок представления.

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

The train suffix OOF contact equation is extremely strong (`~0.006 ft`), and the exact train-copy forensic is also `~0.005 ft`. However, this does not match the real Kaggle public score `7.043`; therefore train `TVT` in the copied wells is not a valid proxy for hidden public suffix labels. Prefix-selected surfaces were `EGFDL`, `ASTNL`, `EGFDL`; oracle-selected surfaces differed again. На момент эксперимента новая submission не была оправдана; позднее public anchor был улучшен до `6.979`.

## Candidate/oracle audit v4

Notebook: [`rogii_candidate_oracle_audit_v1`](rogii_candidate_oracle_audit_v1/rogii_candidate_oracle_audit_v1.ipynb).

Проверена гипотеза, что основная проблема находится не в форме траектории, а в выборе её коэффициентов. Семейство кандидатов:

`TVT_hat = TVT_PS - (Z - Z_PS) + c1*x + c2*x²`.

Результаты random-well OOF на 773 wells:

- continuous quadratic oracle: **4.279138**;
- grid oracle: **5.811822**;
- learned ranker selector: **14.366629**;
- regularized selector: **14.660849**;
- Ridge prior: **14.748959**.

Multi-scale GR cost имеет сильную ordinal-связь с качеством кандидата: median Spearman `0.806735`. Однако learned selector улучшает Ridge лишь на `0.382 ft` и выигрывает только примерно на `19.4%` wells. Вывод: хорошая форма существует, но доступные признаки пока не позволяют надёжно выбрать её параметры.

## Neighbor transfer research v3

Kaggle kernel: [rogii-neighbor-transfer-research-v1](https://www.kaggle.com/code/flexonafft/rogii-neighbor-transfer-research-v1). Локальный notebook: [`rogii_neighbor_transfer_research_v1`](rogii_neighbor_transfer_research_v1/rogii_neighbor_transfer_research_v1.ipynb).

Вместо копирования абсолютного TVT переносился residual относительно геометрического anchor. Random-well OOF:

- `hybrid_md_600`: **13.351120**;
- nested selector: **13.480106**;
- normalized-coordinate hybrid: **13.941482**;
- Ridge prior: **14.748959**.

Это лучшее подтверждённое random-well OOF направление. Но spatial coverage неоднороден: только 10 wells имеют соседа ближе 150 ft, 79 — 150–300 ft, 309 — 300–600 ft, 375 — дальше 600 ft. Проверка допускает близкие reference wells из других random folds и поэтому не доказывает перенос на новый район.

Все 3 public wells являются same-ID train-копиями с self-distance `0`. Ближайшие внешние соседи находятся примерно на `400`, `1112` и `440` ft. Отдельная submission не создавалась: для public ветки она дублировала бы уже известный overlap signal.

## GR-DTW research v3

Kaggle kernel: [rogii-gr-dtw-research-v2](https://www.kaggle.com/code/flexonafft/rogii-gr-dtw-research-v2). Локальный notebook: [`rogii_gr_dtw_research_v1`](rogii_gr_dtw_research_v1/rogii_gr_dtw_research_v1.ipynb).

Исправленный fixed-grid GR-DTW с nested gain дал:

- oracle configuration: **13.330366**;
- nested gated raw: **14.555628** pooled и **14.545582** confirmatory;
- Ridge prior: **14.748959** pooled и **14.723709** confirmatory;
- standalone DTW-конфигурации: **15.091–15.961**.

На exact train-copy pseudo-test frozen `6.979` anchor получил RMSE `0.433752`. Добавление даже 10% GR gate ухудшило его до `0.643558`; 25% — до `1.048057`; полный gate — до `3.276220`. GR несёт слабый дополнительный signal, но опасен как основной путь и не должен корректировать public anchor без well-specific доказательства.

## Monotonic alignment lab v1

Kaggle kernel: [rogii-alignment-lab-v1](https://www.kaggle.com/code/flexonafft/rogii-alignment-lab-v1). Локальный notebook: [`rogii_alignment_lab_v1`](rogii_alignment_lab_v1/rogii_alignment_lab_v1.ipynb).

Проверен multi-scale slope-constrained monotonic DTW на абсолютной TVT-сетке с band `±120 ft`, шагом `2 ft` и ограничением перехода.

- Ridge prior: **14.748959**;
- oracle alignment configuration: **14.390800**;
- self multi-scale: **21.438107**;
- prefix-selected path: **22.625840**;
- typewell/hybrid варианты: **25.832–26.606**.

Prefix-selected path проиграл Ridge во всех spatial evaluation slices. Практический вывод: monotonic-path assumption слишком ограничивает реальную форму или GR cost неоднозначен; «ещё более правильный DTW» не является приоритетной веткой.

## Curvature recoverability lab v1

Kaggle kernel: [rogii-curvature-recoverability-v1](https://www.kaggle.com/code/flexonafft/rogii-curvature-recoverability-v1). Локальный notebook: [`rogii_curvature_recoverability_v1`](rogii_curvature_recoverability_v1/rogii_curvature_recoverability_v1.ipynb).

Это первая полностью обучаемая проверка на outer spatial holdout. Residual относительно геометрического anchor представлен шестью anchored basis components: линейной компонентой и пятью синусами. Коэффициенты предсказывались из legal geometry, prefix/suffix GR landscape и fold-safe neighbor coefficients.

- six-basis oracle: **1.474922**;
- curvature geometry: **19.863795**;
- curvature + neighbor: **20.281388**;
- spatial Ridge: **20.317648**;
- neighbor-вариант выиграл у spatial Ridge в `3/5` folds, но улучшил pooled RMSE только на `0.036260`.

Заранее заданный критерий `RMSE <= 13.0` и победа минимум в `4/5` folds не выполнен. Значение `19–20` нельзя напрямую сравнивать с random-well `13–15`: spatial split исключает из обучения целые районы и является значительно более жёсткой задачей.

Главный вывод: выбранный basis почти идеально описывает target shape, но коэффициенты не переносятся между пространственными областями по текущим признакам. Ограничение находится в определении геологического режима, а не в выразительности формы.

## Консолидированные выводы на 2026-07-20

1. Public leaderboard и unseen-well research — две разные задачи. `6.979` опирается на три same-ID overlaps; улучшение OOF само по себе не обязано улучшать public score.
2. Форма не является главным ограничением: quadratic oracle даёт `4.279`, six-basis oracle — `1.475`.
3. Главный bottleneck — выбор формы или её коэффициентов без suffix TVT.
4. Neighbor residual transfer — самый сильный честный сигнал в random-well OOF (`13.351`), но его преимущество почти исчезает на outer spatial holdout.
5. GR полезен как ranking/confidence feature, но standalone GR alignment и monotonic DTW нестабильны и часто дают cycle/offset errors.
6. Prefix backtest не является надёжным selector для suffix GR path.
7. Same-well formation reconstruction (`~0.005`) объясняет силу overlap-ветки, но не доказывает unseen-well generalization.
8. Следующий эксперимент должен проверять локальную recoverability: пространственно-формационные кластеры, regime classification и только затем prediction коэффициентов внутри режима.
