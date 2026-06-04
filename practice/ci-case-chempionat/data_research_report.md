# Исследование данных ci-case-chempionat

Дата анализа: 2026-06-04. Источник: локальный файл `train_weRmhWx.csv` и решения в `logreg_solution` / `catboost_solution`.

## Executive summary

- Датасет почти наверняка синтетический: 70k строк без пропусков и дублей, 100% случайно выглядящих `user_id`, почти независимые базовые признаки, идеально сбалансированные таргеты около 49-50% и несколько бизнес-неправдоподобных комбинаций.
- Главный практический вывод: **лучше всего обобщает простая регуляризованная логистическая регрессия на 8 исходных признаках**. На 5-fold OOF она дает `micro AUC ~0.67033`; полиномы, сплайны и бустинг в моих проверках не улучшают private-safe метрику.
- Похоже, таргеты сгенерированы как независимые Bernoulli-события от простых линейных/монотонных функций признаков. После логрега остаточные корреляции между продуктами падают примерно до `|corr| < 0.007`, поэтому classifier chains и сложное multi-label моделирование здесь маловероятно помогут.
- Одинаковые public scores у участников не удивляют: сигнал слабый и потолок низкий. Если public split около 14k пользователей, bootstrap по OOF дает 5-95% разброс примерно `0.0045` AUC. Разницы на leaderboard меньше 0.001-0.002 могут быть просто шумом.
- Стратегия для private: выбирать решение по repeated OOF, а не по public; отправлять простую модель с хорошей глобальной калибровкой вероятностей; любые бленды/бустинг принимать только при стабильном OOF-приросте хотя бы `+0.001` micro AUC на нескольких seed.

## 1. Инвентаризация данных

| check | value |
| --- | --- |
| rows | 70,000 |
| columns | 18 |
| features | 8 |
| targets/products | 9 |
| missing cells | 0 |
| duplicate user_id | 0 |
| duplicate feature rows | 0 |
| user_id format | 100% unique 10-char hex strings |
| binary targets | True |

Ключевые признаки синтетики:

- `user_id`: все значения уникальные и выглядят как 10-символьный hex, без явной связи с клиентской историей.
- Нет пропусков, дублей, битых категорий и грязных типов. Для реального банковского датасета это подозрительно чисто.
- `tx_count_30d` хранится как float, но все 100% значений целые.
- Признаки почти независимы друг от друга: максимальная найденная попарная корреляция среди топ-пар всего около `0.0105`.
- `tenure_months` не согласован с возрастом: у `7,139` клиентов (`10.20%`) стаж в месяцах больше, чем число месяцев после 18 лет.

### Распределения признаков

| feature | type | min | mean | p50 | p99 | max | unique | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | numeric | 18.000 | 43.992 | 44.000 | 70.000 | 70.000 | 53 |  |
| income_bucket | categorical/binary | 0.000 | 1.498 |  |  | 3.000 | 4 | 0: 20.2%; 1: 29.8%; 2: 30.0%; 3: 20.0% |
| tenure_months | numeric | 0.000 | 59.798 | 60.000 | 118.000 | 119.000 | 120 |  |
| tx_count_30d | numeric | 2.000 | 14.958 | 15.000 | 25.000 | 36.000 | 33 |  |
| avg_tx_amount | numeric | 1.448 | 75.409 | 54.627 | 356.974 | 1997.316 | 70000 |  |
| digital_activity_score | numeric | 0.001 | 0.285 | 0.265 | 0.704 | 0.903 | 70000 |  |
| has_child | categorical/binary | 0.000 | 0.350 |  |  | 1.000 | 2 | 0: 65.0%; 1: 35.0% |
| is_salary_client | categorical/binary | 0.000 | 0.452 |  |  | 1.000 | 2 | 0: 54.8%; 1: 45.2% |

### Самые заметные корреляции между признаками

| feature_pair | pearson_corr |
| --- | --- |
| age + tx_count_30d | -0.0105 |
| avg_tx_amount + digital_activity_score | +0.0071 |
| income_bucket + avg_tx_amount | -0.0067 |
| digital_activity_score + is_salary_client | -0.0053 |
| tenure_months + avg_tx_amount | -0.0052 |
| income_bucket + is_salary_client | -0.0051 |
| income_bucket + digital_activity_score | +0.0049 |
| tenure_months + is_salary_client | -0.0044 |

Интерпретация: базовые признаки практически независимы. Это похоже на генератор, который сэмплит каждый признак отдельно, а потом генерирует продуктовые таргеты из этих признаков.

## 2. Таргеты и структура multi-label

### Баланс продуктов

| product | positive_rate | positive_count | negative_count |
| --- | --- | --- | --- |
| premium_account | 49.937% | 34956 | 35044 |
| deposit | 49.900% | 34930 | 35070 |
| business_loan | 49.821% | 34875 | 35125 |
| insurance | 49.789% | 34852 | 35148 |
| investment | 49.671% | 34770 | 35230 |
| credit_card | 49.556% | 34689 | 35311 |
| mortgage | 49.240% | 34468 | 35532 |
| p2p_transfer | 49.186% | 34430 | 35570 |
| cashback | 48.963% | 34274 | 35726 |

Все 9 таргетов почти идеально сбалансированы. Это нетипично для реального продуктового портфеля: ипотека, cashback, P2P, премиум-аккаунт и бизнес-кредит обычно не имеют одинаковых базовых частот.

### Сколько продуктов положительны у одного клиента

| positive_products | users | share |
| --- | --- | --- |
| 0 | 398 | 0.57% |
| 1 | 2240 | 3.20% |
| 2 | 6226 | 8.89% |
| 3 | 11474 | 16.39% |
| 4 | 15269 | 21.81% |
| 5 | 15106 | 21.58% |
| 6 | 11098 | 15.85% |
| 7 | 5911 | 8.44% |
| 8 | 1943 | 2.78% |
| 9 | 335 | 0.48% |

Среднее число положительных продуктов на клиента: `4.461` из 9. Распределение похоже на сумму нескольких почти независимых бинарных событий с вероятностью около 0.5, но с небольшой общей зависимостью через сильные признаки (`income_bucket`, `has_child`, `is_salary_client`, `digital_activity_score`).

### Связи между продуктами

| pair | phi_corr | cooccurrence_rate | lift_vs_independent |
| --- | --- | --- | --- |
| mortgage + insurance | +0.164 | 28.627% | 1.168 |
| credit_card + investment | +0.099 | 27.086% | 1.100 |
| credit_card + business_loan | +0.097 | 27.119% | 1.098 |
| investment + business_loan | +0.080 | 26.749% | 1.081 |
| investment + premium_account | +0.077 | 26.730% | 1.078 |
| premium_account + business_loan | +0.073 | 26.710% | 1.074 |
| mortgage + premium_account | +0.071 | 26.353% | 1.072 |
| credit_card + premium_account | +0.059 | 26.233% | 1.060 |
| mortgage + cashback | -0.005 | 23.977% | 0.995 |
| insurance + p2p_transfer | -0.004 | 24.380% | 0.996 |
| mortgage + p2p_transfer | -0.004 | 24.129% | 0.996 |
| p2p_transfer + business_loan | -0.000 | 24.501% | 1.000 |
| cashback + business_loan | +0.000 | 24.406% | 1.000 |
| deposit + p2p_transfer | +0.003 | 24.614% | 1.003 |
| insurance + cashback | +0.003 | 24.464% | 1.004 |
| deposit + cashback | +0.006 | 24.590% | 1.006 |

Самая сильная сырая связь: `mortgage + insurance` (`phi ~ +0.164`, lift `~1.168`). Остальные связи умеренные. Это объясняется не настоящей последовательностью продуктов, а общими признаками: например `has_child` одновременно сильно двигает ипотеку и страховку.

## 3. Где реально есть сигнал

### Top-3 однофакторных AUC по каждому продукту

| product | feature | auc_strength | direction |
| --- | --- | --- | --- |
| business_loan | income_bucket | 0.626 | + |
| business_loan | is_salary_client | 0.597 | + |
| business_loan | has_child | 0.504 | - |
| cashback | digital_activity_score | 0.576 | + |
| cashback | tx_count_30d | 0.505 | + |
| cashback | age | 0.505 | - |
| credit_card | is_salary_client | 0.643 | + |
| credit_card | income_bucket | 0.594 | + |
| credit_card | digital_activity_score | 0.590 | + |
| deposit | income_bucket | 0.553 | + |
| deposit | digital_activity_score | 0.532 | + |
| deposit | tx_count_30d | 0.505 | + |
| insurance | has_child | 0.644 | + |
| insurance | income_bucket | 0.577 | + |
| insurance | tx_count_30d | 0.503 | + |
| investment | income_bucket | 0.626 | + |
| investment | is_salary_client | 0.573 | + |
| investment | digital_activity_score | 0.565 | + |
| mortgage | has_child | 0.709 | + |
| mortgage | income_bucket | 0.621 | + |
| mortgage | age | 0.504 | + |
| p2p_transfer | digital_activity_score | 0.582 | + |
| p2p_transfer | tx_count_30d | 0.506 | + |
| p2p_transfer | tenure_months | 0.504 | + |
| premium_account | income_bucket | 0.676 | + |
| premium_account | digital_activity_score | 0.541 | + |
| premium_account | avg_tx_amount | 0.504 | - |

### Top-3 коэффициентов логрега по каждому продукту

| product | feature | std_logit_coef |
| --- | --- | --- |
| business_loan | income_bucket | +0.491 |
| business_loan | is_salary_client | +0.420 |
| business_loan | has_child | -0.017 |
| cashback | digital_activity_score | +0.272 |
| cashback | tx_count_30d | +0.017 |
| cashback | age | -0.016 |
| credit_card | is_salary_client | +0.634 |
| credit_card | income_bucket | +0.389 |
| credit_card | digital_activity_score | +0.370 |
| deposit | income_bucket | +0.194 |
| deposit | digital_activity_score | +0.116 |
| deposit | tx_count_30d | +0.016 |
| insurance | has_child | +0.647 |
| insurance | income_bucket | +0.309 |
| insurance | avg_tx_amount | +0.013 |
| investment | income_bucket | +0.486 |
| investment | is_salary_client | +0.322 |
| investment | digital_activity_score | +0.249 |
| mortgage | has_child | +1.040 |
| mortgage | income_bucket | +0.567 |
| mortgage | age | +0.021 |
| p2p_transfer | digital_activity_score | +0.297 |
| p2p_transfer | tx_count_30d | +0.022 |
| p2p_transfer | tenure_months | +0.016 |
| premium_account | income_bucket | +0.679 |
| premium_account | digital_activity_score | +0.158 |
| premium_account | has_child | +0.012 |

Практическая картина:

- `income_bucket` — главный общий драйвер для `premium_account`, `business_loan`, `investment`, `mortgage`, `credit_card`, `insurance`, `deposit`.
- `has_child` — почти весь сигнал для `mortgage` и сильный сигнал для `insurance`.
- `is_salary_client` — сильный сигнал для `credit_card`, `business_loan`, `investment`.
- `digital_activity_score` — основной сигнал для `p2p_transfer`, `cashback`, а также заметный для `credit_card` и `investment`.
- `age`, `tenure_months`, `tx_count_30d`, `avg_tx_amount` в основном слабые/шумовые. Их лучше не превращать в агрессивные нелинейные конструкции без жесткой CV-проверки.

## 4. Проверка моделей и обобщения

Метрика из локального `logreg_solution/solution.ipynb`: **pooled/micro ROC-AUC**, один AUC по всей матрице `client × product`. Я считал OOF на train, чтобы не ориентироваться на public leaderboard.

### OOF-сравнение моделей

| model | micro_auc_mean | micro_auc_std | macro_auc_mean | runs |
| --- | --- | --- | --- | --- |
| logreg_ordinal_C1 | 0.670327 | 0.000018 | 0.658411 | 5 |
| logreg_onehot_C1 | 0.670309 |  | 0.658331 | 1 |
| logreg_ordinal_balanced_C1 | 0.670279 |  | 0.658390 | 1 |
| logreg_spline_C0.1 | 0.670079 |  | 0.658037 | 1 |
| logreg_poly2_C0.1 | 0.669819 |  | 0.657802 | 1 |
| hist_gradient_boosting_depth3 | 0.669387 |  | 0.657096 | 1 |
| constant_product_prevalence | 0.503672 |  | 0.500000 | 1 |
| global_long_logreg_product_interactions | 0.670290 |  | 0.658375 | 1 |

Выводы по моделям:

- `logreg_ordinal_C1` стабильно лучший среди проверенных вариантов: `micro AUC mean 0.670327`, std по 5 seed всего `0.000018`.
- One-hot для `income_bucket` почти не меняет результат. Это значит, что ordinal-кодирование случайно хорошо совпадает с генератором: bucket действительно ведет себя монотонно.
- Полиномы, сплайны и shallow gradient boosting ухудшают результат. Это сильный аргумент против усложнения ради public score.
- Глобальная long-format модель `user × product` с product-specific interactions дает почти то же качество (`micro AUC 0.670290`). Она методологически правильна для pooled AUC, но не дает прироста относительно 9 отдельных логрегов.

### OOF AUC логрега по продуктам

| product | auc |
| --- | --- |
| mortgage | 0.77828 |
| credit_card | 0.71388 |
| insurance | 0.68686 |
| premium_account | 0.68621 |
| business_loan | 0.67054 |
| investment | 0.66858 |
| p2p_transfer | 0.58185 |
| cashback | 0.57573 |
| deposit | 0.56357 |

Слабые продукты (`deposit`, `cashback`, `p2p_transfer`) ограничивают общий потолок. Если кто-то резко улучшает public именно на них без repeated OOF-подтверждения, это, скорее всего, подгонка под public.

### Проверка residual target dependency

| pair | raw_phi_corr | residual_corr_after_logreg |
| --- | --- | --- |
| insurance + cashback | +0.003 | +0.0067 |
| cashback + premium_account | +0.014 | +0.0067 |
| deposit + investment | +0.035 | +0.0067 |
| deposit + p2p_transfer | +0.003 | -0.0058 |
| insurance + business_loan | +0.023 | -0.0058 |
| credit_card + premium_account | +0.059 | -0.0057 |
| mortgage + p2p_transfer | -0.004 | -0.0052 |
| insurance + p2p_transfer | -0.004 | -0.0052 |
| credit_card + p2p_transfer | +0.019 | -0.0052 |
| mortgage + investment | +0.047 | -0.0042 |

После простого логрега остаточные корреляции продуктов практически исчезают. Поэтому не стоит ожидать большого выигрыша от classifier chains, label powerset, нейросетей по label-dependency или ручных правил совместной покупки.

## 5. Почему public leaderboard может вводить в заблуждение

### Public-like fold variability на OOF

| sample | users | label_pairs | micro_auc | macro_auc |
| --- | --- | --- | --- | --- |
| user_fold_0 | 14000 | 126000 | 0.66799 | 0.65611 |
| user_fold_1 | 14000 | 126000 | 0.66847 | 0.65672 |
| user_fold_2 | 14000 | 126000 | 0.67156 | 0.65942 |
| user_fold_3 | 14000 | 126000 | 0.67070 | 0.65882 |
| user_fold_4 | 14000 | 126000 | 0.67295 | 0.66106 |

Даже на пяти равных кусках по 14k пользователей `micro AUC` прыгает от `0.66799` до `0.67295`. Это не разные модели, а один и тот же OOF-прогноз на разных подвыборках.

### Bootstrap-оценка шума public split

| public_users | label_pairs | auc_sd | p05 | p50 | p95 | p95_minus_p05 |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | 9000 | 0.00566 | 0.66125 | 0.67094 | 0.67968 | 0.01842 |
| 3000 | 27000 | 0.00315 | 0.66530 | 0.67030 | 0.67593 | 0.01063 |
| 7000 | 63000 | 0.00197 | 0.66714 | 0.67041 | 0.67391 | 0.00678 |
| 14000 | 126000 | 0.00141 | 0.66789 | 0.67030 | 0.67237 | 0.00448 |
| 21000 | 189000 | 0.00107 | 0.66851 | 0.67038 | 0.67208 | 0.00357 |

Интерпретация:

- Если public содержит около `1k` пользователей, нормальный 5-95% разброс оценки может быть около `0.018` AUC.
- Даже при `14k` пользователей 5-95% диапазон около `0.0045` AUC.
- Поэтому разница между участниками в `0.0002`, `0.0005`, даже `0.001` не доказывает, что модель лучше на private.
- Многочисленные submissions позволяют подобрать вариант, который случайно удачен на public, но не обязан быть удачным на hidden private.

## 6. Ответы на твои вопросы

### 1. Какие модели лучше использовать для обобщения

Рекомендуемый стек:

1. **Основной кандидат: 9 независимых LogisticRegression + StandardScaler**, без сложных interactions, без target leakage, с selection по repeated OOF micro AUC.
2. **Альтернатива: global long-format logistic regression** на строках `user × product`, где есть `product_id` и `product_id × feature` interactions. Это удобно, если хочешь прямо оптимизировать общую шкалу pooled metric и контролировать калибровку между продуктами.
3. **LightGBM/CatBoost только как кандидат для бленда**, а не как основа. Принимать их стоит только если repeated OOF показывает стабильный прирост micro AUC, а не только хороший public.
4. **Не ставить в центр нейросети, classifier chains, stacking высокой сложности, полиномы высокого порядка**. На этих данных они с высокой вероятностью учат шум генератора.

### 2. Какой стратегии придерживаться, чтобы иметь шанс на private

- Сделать главным критерием `repeated 5-fold OOF micro AUC`, например 5-10 разных seed. Public использовать только как sanity check.
- Фиксировать один validation protocol и не менять его после каждого public submit.
- Считать improvement реальным только если он стабилен хотя бы на `+0.001` micro AUC и не ухудшает слабые продукты.
- Ограничить число leaderboard-driven submissions. Чем больше перебор по public, тем выше шанс выбрать случайно переобученный вариант.
- Сохранять глобальную калибровку вероятностей. Для pooled AUC важна не только сортировка внутри одного продукта, но и относительная шкала между продуктами.
- Если public score выше OOF на несколько тысячных, относиться к этому как к шуму/переобучению, а не как к доказательству качества.

### 3. Что делать с данными и куда двигаться

Приоритеты:

1. **Уточнить метрику платформы**: если это точно pooled/micro AUC, оптимизировать единую шкалу. Если macro AUC, можно больше думать о per-product качестве. Локальный `logreg_solution` указывает именно pooled/micro.
2. **Сделать надежный OOF harness**: одинаковая функция для train, OOF, финального fit и inference; логировать macro, micro и per-product AUC.
3. **Проверить только малый набор private-safe идей**:
   - logreg `C` grid: `0.03, 0.1, 0.3, 1, 3, 10`;
   - ordinal vs one-hot `income_bucket`;
   - global long-format logreg;
   - легкий blend `0.8 * logreg + 0.2 * calibrated_GBDT`, но только если OOF стабилен;
   - per-label affine calibration по OOF, если есть расхождение между micro и macro.
4. **Не делать агрессивный feature engineering**: признаки уже слишком чистые и независимые. Новые признаки вроде `age/tenure`, binning, high-order interactions, rank transforms стоит оставлять только если они проходят repeated OOF.
5. **Не оптимизировать threshold**: для ROC-AUC пороги не нужны. Нужны качественные вероятности/ранги.

## 7. Практический план на соревнование

1. Оставить `logreg_solution` как сильный baseline.
2. Переписать локальную проверку в один скрипт: repeated folds, `micro_auc`, `macro_auc`, per-product AUC, bootstrap confidence intervals.
3. Сравнить 5-7 строго ограниченных вариантов из раздела выше.
4. Выбрать финальную модель по OOF, а не по public.
5. На всем train обучить выбранную модель и отправить один-два финальных сабмита: основной logreg и, только если подтверждено, легкий OOF-approved blend.

## 8. Итоговая рекомендация

Мое лучшее предположение: hidden private сгенерирован тем же процессом, что и train/public, но public leaderboard достаточно шумный, чтобы провоцировать подгонку. Победная стратегия здесь не в сложной модели, а в дисциплине валидации: **простая линейная модель, единая шкала вероятностей, repeated OOF, минимум public tuning**.

Если private distribution совпадает с train, логрег должен быть очень конкурентным. Если hidden distribution специально сдвинут, то сложный public-overfit пострадает первым; простая регуляризованная модель остается наиболее защищенным вариантом.
