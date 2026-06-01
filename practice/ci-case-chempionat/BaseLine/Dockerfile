FROM python:3.11.2

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHOUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
COPY baseline_model.joblib .
COPY run.py .

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD python /app/run.py --input-path $INPUT_PATH --output-path $OUTPUT_PATH
