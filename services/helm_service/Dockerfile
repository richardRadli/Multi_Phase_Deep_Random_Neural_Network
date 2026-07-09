FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY services/helm_service/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app
WORKDIR /app

COPY --from=builder /install /usr/local

COPY config/ ./config/
COPY nn/ ./nn/
COPY utils/ ./utils/
COPY services/helm_service/ ./services/helm_service/

EXPOSE 8002
CMD ["uvicorn", "services.helm_service.main:app", "--host", "0.0.0.0", "--port", "8002"]