FROM python:3.12-slim

WORKDIR /app

# deps système
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

#copier les fichiers de dépendances (uv)
COPY pyproject.toml uv.lock ./

#installer uv+ deps
RUN pip install --no-cache-dir uv \
 && uv export --format requirements-txt --locked --no-dev -o requirements.txt \
 && pip install --no-cache-dir -r requirements.txt

# copier le cod + artifacts
COPY src ./src

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
