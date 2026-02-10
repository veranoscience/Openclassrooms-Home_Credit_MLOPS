FROM python:3.12-slim

WORKDIR /app

# deps système
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

#installer deps
RUN pip install --no-cache-dir -r requirements.txt

# copier le cod + artifacts
COPY src ./src

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
