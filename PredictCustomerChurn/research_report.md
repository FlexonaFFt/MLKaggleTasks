# SOTA Research: Telecom Customer Churn Prediction

**Kaggle Playground Series — Season 6, Episode 3**
**Метрика оценки: ROC-AUC**
**Дата исследования: 2026-03-24**

---

## 1. Обзор задачи

Задача Kaggle Playground Series S6E3 — бинарная классификация: предсказание оттока клиентов телеком-компании. Метрика — **area under ROC curve (ROC-AUC)** между предсказанной вероятностью и фактическим значением целевой переменной.

### Датасет

Основан на классическом Telco Customer Churn dataset (IBM). Типичные признаки:

| Категория | Признаки |
|-----------|----------|
| Демография | gender, SeniorCitizen, Partner, Dependents |
| Услуги | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Аккаунт | Contract, PaperlessBilling, PaymentMethod, tenure |
| Финансы | MonthlyCharges, TotalCharges |
| Целевая | Churn (Yes/No) |

**Характерные особенности:**
- Дисбаланс классов: ~26-27% churned vs ~73-74% non-churned
- Смесь категориальных и числовых признаков
- Высокая корреляция между tenure и TotalCharges
- Данные Playground Series генерируются синтетически на основе оригинального датасета

---

## 2. Лучшие модели и подходы

### 2.1 Gradient Boosting модели (основа SOTA)

| Модель | Сильные стороны | ROC-AUC (типичный) |
|--------|----------------|---------------------|
| **CatBoost** | Нативная работа с категориальными признаками, устойчивость к переобучению | 0.91-0.93 |
| **LightGBM** | Быстрое обучение, хорошо масштабируется на больших данных, эффективен с несбалансированными данными | 0.90-0.92 |
| **XGBoost** | Сильная регуляризация, предотвращение переобучения | 0.90-0.92 |

**Вывод:** CatBoost часто показывает лучшие результаты на телеком-данных благодаря нативной обработке категориальных фичей и ordered target encoding.

### 2.2 Нейросетевые подходы

- **1D-CNN** — достиг PR-AUC 99.36% на определённых конфигурациях (но это может быть переобучение на конкретном сплите)
- **MLP (Multilayer Perceptron)** — точность 92.28%, precision 0.88, recall 0.82
- **TabNet** — attention-based модель для табличных данных, конкурентоспособна с GBDT

**Вывод:** Нейросети могут быть полезны в ансамблях, но как standalone уступают GBDT на табличных данных такого масштаба.

### 2.3 Классические ML модели

- **Logistic Regression** — полезна как baseline и для stacking
- **Random Forest** — ROC-AUC ~0.84-0.87
- **SVM** — менее конкурентоспособна на этой задаче

---

## 3. Feature Engineering для Telecom Churn

### 3.1 Ключевые трансформации

```python
# Взаимодействие финансовых признаков
df['ChargesPerMonth_per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargesDiff'] = df['MonthlyCharges'] - df['AvgMonthlyCharge']

# Tenure bins (категоризация)
df['TenureBin'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72],
                          labels=['0-1y', '1-2y', '2-3y', '3-4y', '4-5y', '5-6y'])

# Tenure в годах
df['Tenure_years'] = df['tenure'] / 12

# Log-трансформация финансовых признаков (для снижения скошенности)
df['LogMonthlyCharges'] = np.log1p(df['MonthlyCharges'])
df['LogTotalCharges'] = np.log1p(df['TotalCharges'])
```

### 3.2 Interaction Features

```python
# Контракт × Финансы
df['Charges_Contract_Interaction'] = df['MonthlyCharges'] * df['Contract'].map(
    {'Month-to-month': 1, 'One year': 12, 'Two year': 24})

# High Risk Contract (короткий контракт + высокие платежи)
df['HighRiskContract'] = ((df['Contract'] == 'Month-to-month') &
                           (df['MonthlyCharges'] > df['MonthlyCharges'].median())).astype(int)

# Charges × Payment interaction
df['Charges_Payment_Interaction'] = df['MonthlyCharges'] * df['PaymentMethod'].factorize()[0]
```

### 3.3 Service-based Features

```python
# Количество услуг
service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df['NumServices'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

# Weighted Service Score
df['ServiceScore'] = df['NumServices'] / df['MonthlyCharges']

# Наличие security-услуг (OnlineSecurity + TechSupport = protective services)
df['HasProtection'] = ((df['OnlineSecurity'] == 'Yes') |
                        (df['TechSupport'] == 'Yes')).astype(int)

# Streaming без protection — рисковый паттерн
df['StreamingNoProtection'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')) & \
                               (df['HasProtection'] == 0)
```

### 3.4 Самые важные признаки (по SHAP и feature importance)

1. **tenure** — самый сильный предиктор (чем дольше клиент, тем меньше вероятность оттока)
2. **Contract** — month-to-month контракт сильно повышает вероятность оттока
3. **MonthlyCharges** — высокие платежи коррелируют с оттоком
4. **TotalCharges** — коррелирует с tenure
5. **InternetService (Fiber optic)** — Fiber optic клиенты чаще уходят
6. **OnlineSecurity / TechSupport** — наличие снижает отток
7. **PaymentMethod (Electronic check)** — электронный чек коррелирует с оттоком
8. **PaperlessBilling** — paperless billing повышает вероятность оттока

---

## 4. Работа с дисбалансом классов

### 4.1 Сравнение подходов

| Метод | Описание | Эффективность для Churn |
|-------|----------|------------------------|
| **scale_pos_weight / class_weight** | Встроенное взвешивание в GBDT | **Рекомендуется** — простой, эффективный, не требует ресемплинга |
| **Focal Loss** | Усиленное внимание к сложным примерам | **Отлично** — превосходит SMOTE по AUC, особенно при сильном дисбалансе (1-5% minority) |
| **SMOTE** | Генерация синтетических примеров minority class | **Хорошо** — классический подход, но может создавать шумные примеры |
| **SMOTE-ENN** | SMOTE + очистка шумных примеров (Edited Nearest Neighbors) | **Очень хорошо** — до 99.92% accuracy в отдельных исследованиях |
| **Threshold tuning** | Подбор оптимального порога классификации | **Дополнительно** — полезно для precision/recall, но ROC-AUC инвариантен к порогу |

### 4.2 Рекомендации

**Для ROC-AUC метрики:**
- **scale_pos_weight** (XGBoost) или **class_weight='balanced'** — первый выбор
- **Focal Loss** в LightGBM — `objective='binary'` с кастомной loss function
- **SMOTE** — как дополнительный эксперимент, но для ROC-AUC часто не даёт значимого улучшения
- Поскольку ROC-AUC инвариантен к порогу, сам по себе ресемплинг может быть менее критичен, чем для F1/accuracy

---

## 5. Ансамблевые методы

### 5.1 Stacking (рекомендуемый подход)

Stacking — наиболее мощный ансамблевый подход для Kaggle соревнований.

**Архитектура (2-уровневый стекинг):**

```
Level 0 (Base Models):
├── CatBoost (with native categorical handling)
├── LightGBM (with tuned hyperparams)
├── XGBoost (with regularization)
├── Extra Trees / Random Forest
└── (опционально) MLP / TabNet

Level 1 (Meta-Model):
└── Logistic Regression / LightGBM на OOF-предсказаниях Level 0
```

**XCL-Churn Framework** (SOTA подход):
- Интеграция XGBoost + CatBoost + LightGBM
- Soft-voting с оптимизированными весами
- Каждая модель вносит уникальный вклад:
  - XGBoost: регуляризация, предотвращение переобучения
  - CatBoost: нативная обработка категориальных фичей
  - LightGBM: эффективность на несбалансированных данных

### 5.2 Blending (простой и эффективный)

```python
# Weighted Average Blending
final_pred = w1 * catboost_pred + w2 * lgbm_pred + w3 * xgb_pred
# Подбор весов через Optuna или scipy.optimize
```

### 5.3 XMS-Net Architecture

Многоуровневый ансамбль: XGBoost + LightGBM + MLP через 2-уровневый стекинг.
Достигает accuracy ~97% на некоторых конфигурациях.

---

## 6. Рекомендации для нашего пайплайна

### 6.1 Пошаговый план

1. **Baseline** — CatBoost с дефолтными параметрами на сырых данных → оценить ROC-AUC
2. **Feature Engineering** — добавить взаимодействия (charges/tenure, service counts, risk flags)
3. **Оптимизация гиперпараметров** — Optuna для каждой из 3 моделей (CatBoost, LightGBM, XGBoost)
4. **Ансамбль** — Weighted Blending или Stacking
5. **Post-processing** — Rank averaging для финального сабмита

### 6.2 Гиперпараметрическая оптимизация (Optuna)

Рекомендуемые пространства поиска:

**XGBoost:**
```python
params = {
    'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
}
```

**LightGBM:**
```python
params = {
    'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
    'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    'is_unbalance': True,
}
```

**CatBoost:**
```python
params = {
    'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
    'depth': trial.suggest_int('depth', 4, 10),
    'iterations': trial.suggest_int('iterations', 100, 1500),
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
    'bagging_temperature': trial.suggest_float('bagging_temp', 0.0, 1.0),
    'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
    'auto_class_weights': 'Balanced',
}
```

**Рекомендация:** ~50-100 trials для каждой модели достаточно для поиска хороших гиперпараметров.

### 6.3 Кросс-валидация

- **StratifiedKFold(n_splits=5)** — стандарт для бинарной классификации с дисбалансом
- **RepeatedStratifiedKFold(n_splits=5, n_repeats=3)** — для более стабильной оценки
- Использовать OOF (out-of-fold) предсказания для стекинга

### 6.4 Encoding стратегия

- **CatBoost:** передавать категориальные признаки напрямую (cat_features)
- **LightGBM/XGBoost:** Label Encoding или Target Encoding (осторожно с утечкой!)
- One-Hot Encoding — для признаков с малым числом категорий (<5)

### 6.5 Финальный ансамбль (рекомендация)

```python
# Вариант 1: Simple Weighted Blending
final = 0.4 * catboost_oof + 0.3 * lgbm_oof + 0.3 * xgb_oof

# Вариант 2: Stacking с Logistic Regression
from sklearn.linear_model import LogisticRegression
meta = LogisticRegression()
meta.fit(np.column_stack([cat_oof, lgb_oof, xgb_oof]), y_train)
final = meta.predict_proba(np.column_stack([cat_test, lgb_test, xgb_test]))[:, 1]

# Вариант 3: Rank averaging (стабильнее при разных масштабах)
from scipy.stats import rankdata
final = (rankdata(cat_pred) + rankdata(lgb_pred) + rankdata(xgb_pred)) / 3
```

---

## 7. Ожидаемые метрики (ROC-AUC бенчмарки)

| Подход | Ожидаемый ROC-AUC | Комментарий |
|--------|-------------------|-------------|
| Logistic Regression (baseline) | 0.82-0.85 | Простой baseline |
| Random Forest | 0.84-0.87 | Без тюнинга |
| Single XGBoost (tuned) | 0.90-0.92 | С Optuna |
| Single LightGBM (tuned) | 0.90-0.92 | С Optuna |
| Single CatBoost (tuned) | 0.91-0.93 | С Optuna, нативные категории |
| Ensemble (Blending 3 GBDT) | 0.92-0.94 | Weighted average |
| Stacking (GBDT + Meta) | 0.93-0.95 | 2-level stacking |
| Stacking + FE + Tuning | 0.94-0.96+ | Полный пайплайн |

**Для Kaggle Playground Series:** top решения обычно достигают ROC-AUC **0.90-0.93+** на public leaderboard. С хорошим feature engineering и ансамблем можно целиться в **top 10%**.

**Примечание:** Playground Series данные синтетические, поэтому абсолютные значения ROC-AUC могут отличаться от реальных телеком-данных. Важнее относительное улучшение и стабильность CV-скора.

---

## Источники

- [Kaggle Playground Series S6E3 — Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3)
- [Explainable churn prediction with SHAP analysis (Springer, 2026)](https://link.springer.com/article/10.1007/s44163-026-00983-0)
- [Telecom churn prediction — 93.61% AUC](https://github.com/NakibIheb20/telecom-churn-prediction)
- [Mitigating class imbalance with ensemble methods and SMOTE (Nature, 2025)](https://www.nature.com/articles/s41598-025-01031-0)
- [Hybrid framework for churn prediction (Journal of Big Data, 2024)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00922-9)
- [Comparative Study of XGBoost, LightGBM, CatBoost for churn](https://www.researchgate.net/publication/397440582)
- [Adaptive ensemble learning for churn (arXiv, 2024)](https://arxiv.org/pdf/2408.16284)
- [Combining predictive accuracy and interpretability (Nature, 2025)](https://www.nature.com/articles/s41598-025-34624-w)
- [Hyperparameter Optimization for churn (MDPI, 2023)](https://www.mdpi.com/2227-7080/11/6/167)
- [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Telco Churn Prediction — Feature Engineering (Kaggle notebook)](https://www.kaggle.com/code/mechatronixs/telco-churn-prediction-feature-engineering-eda)
