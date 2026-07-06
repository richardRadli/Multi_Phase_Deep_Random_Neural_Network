FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY services/training_and_evaluation/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app
WORKDIR /app

COPY --from=builder /install /usr/local

COPY config/ ./config/
COPY nn/ ./nn/
COPY utils/ ./utils/

COPY services/training_and_evaluation/ ./services/training_and_evaluation/

EXPOSE 8001
CMD ["uvicorn", "services.training_and_evaluation.main:app", "--host", "0.0.0.0", "--port", "8000"]