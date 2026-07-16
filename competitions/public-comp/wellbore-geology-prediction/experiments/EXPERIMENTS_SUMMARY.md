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

## Сохранено вне этого архива

`datasets/` не удалён. Он содержит локальный Kaggle dataset и нужен для дальнейшего обучения. Остальные старые experiment artifacts удалены после создания этого summary.
