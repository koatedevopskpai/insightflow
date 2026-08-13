# Insightflow serving image.
#
# Design: the image ships the application code + dependencies only. The
# trained model and its registry (mlflow.db + mlruns/) are mounted at
# runtime via docker-compose, so this image never goes stale when a new
# model is promoted — you rebuild the registry, not the container.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (excluding versioned data, tests, and runtime state).
COPY features.py data.py model.py serve.py ./

# The frozen model directory is mounted at runtime (see docker-compose.yml)
# so the image stays model-agnostic.
ENV MLFLOW_MODEL_URI=/models/production

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
