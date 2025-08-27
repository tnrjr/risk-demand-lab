
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (add gcc if you need to compile libs)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

# Copy project metadata and install
COPY pyproject.toml /app/pyproject.toml
RUN pip install --upgrade pip && pip install -e ".[dev]"

# Copy source
COPY src /app/src

# Default MLflow endpoint (overridable via env)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
