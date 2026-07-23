# 1. FÁZIS: Builder
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY services/dataset_operations_service/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 2. FÁZIS: Final
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app
ENV DATASET_ROOT=/app/datasets
ENV STORAGE_ROOT=/app/storage
WORKDIR /app

COPY --from=builder /install /usr/local

COPY config/ ./config/
COPY utils/ ./utils/

COPY services/dataset_operations_service/ ./services/dataset_operations_service/

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "services.dataset_operations_service.api.api:app", "--host", "0.0.0.0", "--port", "8000"]