
# ========= Semana 1 – Fundação =========
PY := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
MLFLOW := $(VENV)/bin/mlflow
PRECOMMIT := $(VENV)/bin/pre-commit
PYTEST := $(VENV)/bin/pytest

.PHONY: venv install precommit test lint format train api mlflow up down clean

venv:
	@echo ">> Creating virtualenv at $(VENV)"
	$(PY) -m venv $(VENV)
	@echo ">> Upgrading pip"
	$(PIP) install --upgrade pip
	@echo ">> Installing project + dev deps"
	$(PIP) install -e ".[dev]"
	@echo ">> Installing pre-commit hooks"
	$(PRECOMMIT) install
	@echo ">> Done. Activate with: source $(VENV)/bin/activate"

test:
	@echo ">> Running tests"
	$(PYTEST) -q

lint:
	@echo ">> Linting with ruff"
	$(VENV)/bin/ruff check 

format:
	@echo ">> Formatting with black"
	$(VENV)/bin/black .

train:
	@echo ">> Training demo model (logs to MLflow)"
	MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-file:./mlruns} \
	$(PYTHON) -m src.train.train_credit

api:
	@echo ">> Starting API (FastAPI + Uvicorn) at http://127.0.0.1:8000"
	MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-file:./mlruns} \
	$(UVICORN) src.app.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	@echo ">> Launching MLflow UI at http://127.0.0.1:5000"
	$(MLFLOW) ui --host 0.0.0.0 --port 5000

up:
	@echo ">> docker compose up (API + MLflow)"
	docker compose up -d --build

down:
	@echo ">> docker compose down"
	docker compose down

clean:
	@echo ">> Cleaning caches and artifacts"
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info mlruns
