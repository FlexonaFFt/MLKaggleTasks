# Stage 2: Robust Modeling And Leaderboard Strategy

Дата анализа: 2026-06-05. Источник: `train_weRmhWx.csv`, текущие `logreg_solution` / `catboost_solution`, локальные OOF-прогоны без сохранения промежуточных CSV/JSON.

## Executive summary

- **Лучший private-safe сценарий остается простым:** 9 независимых `LogisticRegression + StandardScaler`, ordinal-кодирование `income_bucket`, без `class_weight`, без агрессивного feature engineering. `C` почти не влияет; `C=0.03` и `C=1` дают практически одинаковый repeated OOF.
- **Single-seed победитель не равен стратегическому победителю.** На одном seed `strong_only` модель дала `micro AUC 0.670339`, но на repeated OOF полный логрег чуть стабильнее: `0.670327` против `0.670323`. Разница `0.000004` меньше любого разумного шума оценки.
- **Бленды не доказали реального преимущества.** Лучший найденный blend уровня `logreg + onehot` давал около `0.670328`, то есть прирост примерно `+0.000005` к базовому логрегу. Это на порядки меньше шума public leaderboard.
- **Подгонка под public выглядит опаснее, чем полезнее.** В первом исследовании bootstrap показал, что при public около 14k пользователей 5-95% диапазон public AUC может быть около `0.0045`, а различия между нашими лучшими моделями находятся в диапазоне `0.000005-0.00004`.
- **Private выборку локально найти нельзя.** В репозитории нет `test.csv`, `sample_submission`, public/private split-файлов или признаков hidden выборки. Docker-интерфейс только получает скрытый `--input-path` во время оценки.

## 1. Что проверялось на втором этапе

Цель второго этапа: не повторять EDA, а выбрать стратегию, которая с наибольшей вероятностью переживет hidden private.

Проверенные группы гипотез:

| group | candidates | purpose | result |
| --- | --- | --- | --- |
| Regularization | `LogisticRegression C=0.003..30` | понять, есть ли переобучение в линейной модели | `C` почти не влияет |
| Feature ablation | all features, strong-only, weak-only | понять, где находится сигнал | сигнал почти весь в 4 strong-признаках |
| Encoding | ordinal vs one-hot `income_bucket` | проверить, не ломает ли ordinal bucket | ordinal не хуже one-hot |
| Nonlinear transforms | quantile, polynomial degree 2 | проверить простую нелинейность | ухудшают OOF |
| Tree model | `HistGradientBoostingClassifier`; существующий `catboost_solution` | проверить нелинейный baseline | хуже логрега / не main candidate |
| Long-format model | `user x product` + product interactions | напрямую моделировать pooled AUC | почти равно логрегу, без прироста |
| Calibration | cross-fit product-specific affine calibration | улучшить общую шкалу между продуктами | ухудшает micro AUC |
| Blends | logreg + onehot / HGB / long / calibrated | найти осторожный ensemble | прирост микроскопический, не надежный |

## 2. Single-seed model search

Все значения ниже посчитаны как 5-fold OOF на `seed=42`, pooled/micro ROC-AUC и macro ROC-AUC по 9 продуктам.

| model | micro AUC | macro AUC | interpretation |
| --- | ---: | ---: | --- |
| `logreg_strong_only_C1` | 0.670339 | 0.658281 | лучший на одном seed, но не лучший repeated |
| `logreg_ordinal_C30` | 0.670323 | 0.658390 | практически то же, что C=1 |
| `logreg_ordinal_C10` | 0.670323 | 0.658390 | практически то же, что C=1 |
| `logreg_ordinal_C1` | 0.670323 | 0.658390 | текущий сильный baseline |
| `logreg_ordinal_C0.03` | 0.670323 | 0.658391 | чуть более регуляризованный аналог |
| `logreg_onehot_C1` | 0.670309 | 0.658331 | не дает прироста |
| `global_long_logreg_C1` | 0.670288 | 0.658374 | методологически красиво, но хуже |
| `logreg_balanced_C1` | 0.670279 | 0.658390 | class_weight портит pooled калибровку |
| `logreg_poly2_C0.1` | 0.669819 | 0.657802 | interactions ловят шум |
| `hist_gradient_boosting` | 0.669387 | 0.657096 | нелинейная модель хуже |
| `logreg_quantile_C1` | 0.666959 | 0.656020 | transform явно вреден |
| `logreg_weak_only_C1` | 0.503571 | 0.501270 | weak-признаки почти не несут сигнала |

Вывод: single-seed таблица соблазняет выбрать `strong_only`, но это опасно. Разница между `strong_only` и полным логрегом всего `0.000016`, а на repeated OOF она исчезает.

Примечание по CatBoost: существующий `catboost_solution` из первого исследования уже показывал `macro OOF AUC ~0.652`, что ниже логрега. Полный повторный CatBoost OOF на втором этапе был остановлен как слишком дорогой для текущей цели: деревообразные кандидаты уже не проходили главный критерий private-safe superiority.

## 3. Repeated OOF stability

Repeated OOF на seed `7, 13, 42, 123, 2026`.

| model | runs | micro mean | micro std | micro min | micro max | macro mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `logreg_ordinal_C1` | 5 | 0.670327 | 0.000018 | 0.670304 | 0.670355 | 0.658411 |
| `logreg_ordinal_C0.03` | 5 | 0.670327 | 0.000018 | 0.670304 | 0.670355 | 0.658411 |
| `logreg_strong_only_C1` | 5 | 0.670323 | 0.000013 | 0.670305 | 0.670339 | 0.658262 |
| `logreg_onehot_C1` | 5 | 0.670311 | 0.000019 | 0.670294 | 0.670343 | 0.658342 |
| `global_long_C1` | 5 | 0.670288 | 0.000017 | 0.670261 | 0.670309 | 0.658337 |

Практический вывод:

- `C=0.03` и `C=1` эквивалентны в пределах численного шума.
- `strong_only` можно держать как conservative/risk альтернативу, но она не доказала superiority.
- `onehot` и `global_long` методологически нормальны, но OOF говорит не выбирать их как main.

## 4. Calibration and blending

### Calibration

| candidate | micro AUC | macro AUC | decision |
| --- | ---: | ---: | --- |
| raw `logreg_C1` OOF | 0.670323 | 0.658390 | keep |
| cross-fit product affine calibration | 0.670278 | 0.658304 | reject |

Калибровка вероятностей между продуктами не улучшила pooled AUC. Похоже, базовый логрег уже достаточно хорошо держит общую шкалу.

### Blend grid

| blend | best weight | micro AUC | delta vs base | decision |
| --- | ---: | ---: | ---: | --- |
| `logreg_C1 + onehot` | 0.35 | 0.670328 | +0.000005 | too small |
| `logreg_C1 + HGB` | 0.05 | 0.670326 | +0.000003 | too small |
| `logreg_C1 + global_long` | 0.20 | 0.670325 | +0.000002 | too small |
| `logreg_C1 + poly2` | 0.00 | 0.670323 | +0.000000 | reject |
| `logreg_C1 + calibrated` | 0.00 | 0.670323 | +0.000000 | reject |
| `logreg_C1 + quantile` | 0.00 | 0.670323 | +0.000000 | reject |

Эти приросты нельзя считать реальными. Если public leaderboard показывает, что такой blend лучше на `0.0005`, это почти наверняка public noise, а не подтверждение качества.

## 5. Обобщение против подгонки под public

Из первого исследования:

| pseudo-public users | label pairs | AUC std | p05 | p50 | p95 | p95-p05 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 9,000 | 0.005659 | 0.661253 | 0.670940 | 0.679675 | 0.018423 |
| 3,000 | 27,000 | 0.003147 | 0.665298 | 0.670295 | 0.675929 | 0.010631 |
| 7,000 | 63,000 | 0.001973 | 0.667137 | 0.670412 | 0.673914 | 0.006777 |
| 14,000 | 126,000 | 0.001405 | 0.667892 | 0.670296 | 0.672368 | 0.004477 |
| 21,000 | 189,000 | 0.001074 | 0.668510 | 0.670384 | 0.672082 | 0.003571 |

Сравнение масштаба:

| quantity | magnitude |
| --- | ---: |
| best stable model delta vs baseline | около `0.000000-0.000004` |
| best blend delta vs baseline | около `0.000005` |
| single-seed strong-only delta vs baseline | около `0.000016` |
| public noise if public ~14k users | около `0.0045` for p95-p05 |

Вывод: public leaderboard на такой задаче легко переоценивает случайные микро-изменения. Подгонять модель под public имеет смысл только если public-прирост крупный и одновременно подтвержден repeated OOF. Для текущих кандидатов такого нет.

## 6. Можно ли найти private выборку

Короткий ответ: **нет, по локальным данным и коду private выборку найти нельзя**.

Проверка локальных файлов:

| check | result |
| --- | --- |
| `test.csv` / `sample_submission` / public/private split files | не найдены |
| найденные `test_error.tsv` | это CatBoost eval logs, не тестовая выборка |
| `run.py` interface | получает только `--input-path` и пишет `--output-path` |
| private labels | недоступны |
| hidden test features | передаются контейнеру только во время оценки |

Что можно делать легально и полезно:

- Проверять, нет ли leakage в локальных файлах, `user_id`, порядке строк, Docker-интерфейсе.
- Сравнивать public score с локальным OOF как sanity check.
- Вести submission log: model, OOF micro, OOF macro, public score, комментарий.
- Отправить один robust candidate и, если разрешено финальными слотами, один controlled-risk candidate.

Что не стоит делать:

- Агрессивно probing-ить public множеством сабмитов и выбирать случайного победителя.
- Пытаться восстановить private labels или hidden split через интерфейс платформы.
- Использовать public score как основной критерий при различиях меньше `0.001-0.002`.

## 7. Лучший сценарий публикации решения

### Main submission

Рекомендация: **полный ordinal logistic regression**.

| setting | value |
| --- | --- |
| model | 9 independent `LogisticRegression` |
| features | all 8 original features |
| preprocessing | `StandardScaler` |
| `income_bucket` | ordinal integer |
| `class_weight` | `None` |
| `C` | `0.03` or `1.0`; practically equivalent |
| selection metric | repeated OOF pooled/micro AUC |
| final fit | train on all rows |

Если менять текущий `logreg_solution`, я бы выбрал `C=0.03` как чуть более регуляризованный вариант. Если не хочется трогать уже работающий pipeline, `C=1.0` тоже нормален: repeated OOF не различает их.

### Controlled-risk alternative

Можно держать один альтернативный финальный сабмит:

| candidate | why | risk |
| --- | --- | --- |
| `strong_only` logreg | убирает почти шумовые weak-признаки | может проиграть, если private генератор использует слабые признаки так же, как train |
| `0.65 * logreg + 0.35 * onehot` | лучший single-seed blend | прирост слишком мал, легко public-overfit |

Я бы не делал их main submission без независимого подтверждения public + repeated OOF.

## 8. Decision rule

Использовать такое правило отбора:

1. Если модель улучшает public, но не улучшает repeated OOF, **не выбирать как main**.
2. Если модель улучшает repeated OOF меньше чем на `0.0005`, считать ее equivalent, а не better.
3. Если модель ухудшает weak products (`deposit`, `cashback`, `p2p_transfer`) без общего repeated OOF прироста, reject.
4. Если public gain меньше `0.001-0.002`, считать его шумом.
5. Финальный выбор: robust OOF winner, а не public winner.

## 9. Итог

Следующий этап подтверждает вывод первого исследования: задача не про сложное моделирование, а про дисциплину валидации. Данные похожи на простой синтетический генератор с линейно-монотонными зависимостями. Сложные модели, transforms, calibration и blends не дают устойчивого прироста.

Лучший шанс на hidden private: **не подгонять public, а публиковать простой регуляризованный логрег с repeated OOF-подтверждением**. Private выборку из локального проекта найти нельзя; можно только строить модель, которая лучше всего обобщает предполагаемый генератор.
