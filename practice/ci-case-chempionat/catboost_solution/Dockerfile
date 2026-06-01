FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY run.py .
COPY models ./models

CMD python /app/run.py --input-path $INPUT_PATH --output-path $OUTPUT_PATH
