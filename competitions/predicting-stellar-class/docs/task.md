# Predicting Stellar Class

Источник: [Kaggle Playground Series - Season 6, Episode 6](https://www.kaggle.com/competitions/playground-series-s6e6/overview)  
Актуально на: 2026-06-01

## Краткое условие

Нужно построить модель многоклассовой классификации, которая по табличным признакам астрономического объекта предсказывает его звездный класс.

Целевая переменная: `class`.

Возможные классы:

- `GALAXY`
- `STAR`
- `QSO`

Задача относится к Kaggle Playground Series: датасет синтетически сгенерирован на основе реального датасета Stellar Classification Dataset SDSS17. Распределения признаков близки к исходному датасету, но не совпадают с ним полностью.

## Данные

Файлы соревнования:

- `train.csv` - обучающая выборка, содержит целевой столбец `class`.
- `test.csv` - тестовая выборка, для нее нужно предсказать `class`.
- `sample_submission.csv` - пример файла отправки в корректном формате.

Идентификатор строки: `id`.

Количество строк в скрытом решении Kaggle: `247435`.

## Формат отправки

Для каждого `id` из тестовой выборки нужно предсказать один из трех классов в столбце `class`.

```csv
id,class
577347,STAR
577348,GALAXY
577349,STAR
```

## Метрика

Сабмиты оцениваются по `balanced accuracy`.

Метрика усредняет полноту по классам:

```text
balanced_accuracy = mean(recall по каждому классу)
```

Практический смысл: важно хорошо предсказывать каждый класс, а не только самый частый. Если классы несбалансированы, обычная accuracy может быть обманчивой, поэтому при локальной валидации лучше смотреть именно `balanced_accuracy_score`.

## Важные параметры соревнования

- Старт: 2026-06-01.
- Финальный дедлайн: 2026-06-30 23:59 UTC.
- Entry deadline: совпадает с финальным дедлайном.
- Team merger deadline: совпадает с финальным дедлайном.
- Public leaderboard: 20% тестовой выборки.
- Максимум сабмитов: 10 в день.
- Финальных сабмитов для оценки: до 2.
- Максимальный размер команды: 3 участника.
- Лицензия данных: CC BY 4.0.
- Призы: Kaggle merchandise за 1-3 места.

## Информация для решения

Базовый план:

1. Загрузить `train.csv`, `test.csv`, `sample_submission.csv`.
2. Проверить размеры, типы столбцов, пропуски, дубликаты `id`, распределение `class`.
3. Отделить `id` и `class`; обучать модель только на признаках.
4. Использовать стратифицированную кросс-валидацию, например `StratifiedKFold`.
5. Валидировать модели через `balanced_accuracy_score`.
6. Сравнить несколько сильных табличных моделей: `CatBoostClassifier`, `LGBMClassifier`, `XGBClassifier`, `RandomForest` или `ExtraTrees`.
7. Если классы несбалансированы, проверить `class_weight`, oversampling или настройку весов.
8. Усреднить предсказания нескольких folds или нескольких моделей, затем выбрать класс с максимальной вероятностью.
9. Сохранить `submission.csv` строго в формате `id,class`.

Что особенно важно:

- Не использовать `id` как обычный признак без проверки: он может не нести полезного сигнала и иногда ухудшает обобщение.
- Делать стратифицированную валидацию, чтобы каждый fold содержал все классы.
- Следить за метрикой по каждому классу: низкий recall одного класса будет напрямую бить по balanced accuracy.
- Для бустингов сначала построить простой baseline, затем добавлять подбор гиперпараметров и ансамбли.
- Проверить, совпадает ли набор колонок в `train.csv` и `test.csv`, кроме отсутствующего `class`.

Минимальный локальный baseline:

```python
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier

train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")

X = train.drop(columns=["id", "class"])
y = train["class"]
X_test = test.drop(columns=["id"])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, valid_idx in cv.split(X, y):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    scores.append(balanced_accuracy_score(y_valid, pred))

print(scores, sum(scores) / len(scores))
```

## Полезные ссылки

- [Страница соревнования](https://www.kaggle.com/competitions/playground-series-s6e6/overview)
- [Описание метрики balanced accuracy в scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [Исходный датасет, вдохновивший соревнование](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/code)
