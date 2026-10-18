# Insightflow

An end-to-end **MLOps lifecycle** project: a production-style machine learning pipeline on the California Housing dataset with data versioning, experiment tracking, Bayesian hyperparameter optimization, model registry promotion, and a containerized inference API.

The repository demonstrates the engineering practices that separate an ML project from an ML *product* — reproducibility, auditability, and a guarded path from experiment to production.

## Pipeline Overview

```
┌──────────────┐   ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐
│ Data         │──▶│ Feature      │──▶│ Optuna sweep   │──▶│ MLflow Model     │
│ versioning   │   │ engineering  │   │ (Bayesian TPE) │   │ Registry         │
│ (DVC)        │   │ features.py  │   │ optimize.py    │   │ (gated promote)  │
└──────────────┘   └──────────────┘   └────────────────┘   └────────┬─────────┘
                                                                    ▼
                                                          ┌──────────────────┐
                                                          │ FastAPI serving  │
                                                          │ + Docker         │
                                                          │ (serve.py)       │
                                                          └──────────────────┘
```

**Stage 1 — Data versioning (DVC).** The train/test CSVs are tracked by DVC with content-addressed hashing (`*.dvc` files in git, blobs in a remote store). Any experiment can be reproduced against the exact data it was trained on.

**Stage 2 — Feature engineering (`features.py`).** Derived features (`RoomsPerHousehold`, `BedroomsPerRoom`, `PopulationPerHousehold`, `LogMedInc`) live in one shared module used by both training and serving — structurally preventing train/serve skew.

**Stage 3 — Bayesian optimization (`optimize.py`).** Optuna's TPE sampler explores `n_estimators` / `max_depth` over the shared training core, logging every trial as an MLflow run.

**Stage 4 — Model registry (`optimize.py`).** The best candidate is registered and promoted to Production — but only if it beats an absolute quality gate (`--min-rmse`) *and* the incumbent Production model. This is the ML analogue of a CI merge gate.

**Stage 5 — Serving (`serve.py`).** A FastAPI app loads the frozen Production model once at startup and exposes a validated `/predict` endpoint, containerized with Docker.

## Project Structure

```
insightflow
├── features.py           # Shared feature engineering (train/serve contract)
├── data.py               # Canonical dataset loading
├── model.py              # Pipeline + training core
├── prepare_data.py       # Build + engineer + split the dataset
├── train.py              # Single training run (argparse + MLflow autolog)
├── optimize.py           # Optuna Bayesian sweep + registry promotion
├── export_model.py       # Freeze Production model into models/production
├── serve.py              # FastAPI inference endpoint
├── Dockerfile            # Serving image (model mounted at runtime)
├── docker-compose.yml    # API + frozen model volume
├── tests/                # Hermetic pytest suite
├── .github/workflows/    # CI: lint, test, data-change training
└── examples/             # Standalone LLM demo (appendix)
```

## Getting Started

### Prerequisites

- Python 3.11+ (developed on 3.14)
- Docker (for the serving container)

### Setup

```bash
git clone https://github.com/koatedevopskpai/insightflow.git
cd insightflow
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

## Running the Pipeline

```bash
# 1. Prepare data (feature engineering + split)
python prepare_data.py

# 2. Train a single model run (logged to MLflow)
python train.py --n-estimators 100 --max-depth 10

# 3. Bayesian hyperparameter search + gated promotion
python optimize.py --n-trials 20 --min-rmse 0.75

# 4. Inspect runs
mlflow ui          # http://localhost:5000

# 5. Freeze the Production model and serve it
python export_model.py
docker compose up --build
curl http://localhost:8000/health
```

### Calling the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc":8.3,"HouseAge":30,"AveRooms":6.1,"AveBedrms":1.1,
       "Population":1000,"AveOccup":3.2,"Latitude":37.7,"Longitude":-122.2}'
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Data Versioning with DVC

The project uses a **Google Cloud Storage (GCS) remote** (`gs://insightflow-dvc/dvc`). GitHub Actions authenticates via **Workload Identity Federation** (OIDC) — no long-lived service-account keys are stored in the repo. CI pulls versioned data before training:

```bash
dvc pull          # restore data from the GCS remote
dvc push          # upload versioned data to the GCS remote
```

### Quality Gates

- **Lint & tests** run on every push/PR (GitHub Actions): `ruff check .` + `pytest`.
- **Training** runs only when data or model code changes (`*.csv.dvc`, `prepare_data.py`, `optimize.py`, `model.py`) or on manual dispatch.
- **Promotion** requires the candidate to beat both `--min-rmse` and the incumbent Production model.

## Key Architectural Decisions

1. **Shared feature engineering (`features.py`)** — the single biggest cause of silent production failures in ML is train/serve skew. Training and the API both call `engineer_features`, so the transform can never diverge.
2. **Application factory (`create_app`)** — lets tests inject a model without touching the registry, keeping the suite hermetic and fast.
3. **Frozen model artifact for Docker** — MLflow registry artifact URIs are absolute host paths, so a container can't resolve a registry produced on another machine. `export_model.py` freezes the Production model into a plain directory that's mounted into the container, decoupling serving from the authoring environment.
4. **Gated promotion** — a candidate never silently replaces a deployed model; it must clear an absolute RMSE threshold and beat the incumbent.
5. **Config via environment** — `MLFLOW_TRACKING_URI` / `MLFLOW_MODEL_URI` mean the same code runs against local SQLite or a remote tracking server with zero edits.

## Testing

```bash
pytest          # 13 hermetic tests (no data files, no registry, no network)
```

## Example (Appendix)

`examples/dad_joke_app/` contains the original GPT-4o-mini terminal app — kept as a separate, self-contained demo unrelated to the MLOps pipeline.

## License

This project is open source. See the `LICENSE` file for details (if applicable).

## Acknowledgements

Built with [scikit-learn](https://scikit-learn.org/), [MLflow](https://mlflow.org/), [DVC](https://dvc.org/), [Optuna](https://optuna.org/), [FastAPI](https://fastapi.tiangolo.com/), and [Docker](https://www.docker.com/).


## Claim review — 18 October 2026

See [verification and limits](evidence/W5-04.md) for the source revision, claim
classification and observed CI status. Wording corrections do not establish a
new deployment, passing test suite or production operation.
