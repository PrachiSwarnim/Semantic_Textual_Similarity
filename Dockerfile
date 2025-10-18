FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && pip install --no-cache-dir --default-timeout=120 -r requirements.txt \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY app.py .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]