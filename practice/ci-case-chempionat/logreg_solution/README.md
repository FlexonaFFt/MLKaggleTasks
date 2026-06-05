# CI Case LogReg submission

Финальное решение использует общий `StandardScaler` и девять отдельных
`LogisticRegression`, по одной модели на каждый продукт. Выходные вероятности
записываются без заголовка в следующем порядке:

1. `credit_card`
2. `mortgage`
3. `deposit`
4. `investment`
5. `insurance`
6. `p2p_transfer`
7. `cashback`
8. `premium_account`
9. `business_loan`

Конфигурация модели: L2, `C=0.03`, `class_weight=None`, все восемь признаков.

## Переобучение

Запускайте обучение в окружении с версиями из `requirements.txt`:

```bash
python train.py --data ../train_weRmhWx.csv --out ./models
```

## Сборка для платформы

Серверная платформа обычно использует `linux/amd64`. На Apple Silicon
архитектуру нужно указывать явно:

```bash
docker build --platform linux/amd64 -t ci-case-logreg:1.0 .
```

## Локальная проверка

В каталоге `/tmp/ci-case-test` должны находиться `input.csv` и место для
`output.csv`:

```bash
docker run --rm --platform linux/amd64 \
  -e INPUT_PATH=/data/input.csv \
  -e OUTPUT_PATH=/data/output.csv \
  -v /tmp/ci-case-test:/data \
  ci-case-logreg:1.0
```

Результат должен содержать столько же строк, сколько входной файл, и ровно
девять колонок вероятностей без заголовка.

## Публикация

```bash
docker login
docker tag ci-case-logreg:1.0 USERNAME/ci-case-logreg:1.0
docker push USERNAME/ci-case-logreg:1.0
```

В интерфейсе соревнования укажите:

```text
USERNAME/ci-case-logreg:1.0
```
