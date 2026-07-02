FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt



FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg


ENV PROJECT_ROOT=/app
ENV DATASET_ROOT=/app/datasets
ENV STORAGE_ROOT=/app/storage


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY --from=builder /install /usr/local

COPY . .


RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]