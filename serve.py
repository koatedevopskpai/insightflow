"""FastAPI inference endpoint for the California Housing model.

Loads the Production-registered model once at startup (not per request)
and exposes a typed /predict endpoint plus a liveness probe.

Architectural decisions:
- Model loads once at process start; per-request loading would serialize
  inference behind filesystem reads and multiply cold-start latency.
- Feature engineering runs through the same features.engineer_features
  used at training time, so the API can never diverge from training
  (no train/serve skew).
- A factory (create_app) lets tests inject a model without hitting the
  registry or network.

Usage:
    uvicorn serve:app --reload --port 8000
"""

import os
from typing import Any

import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import RAW_FEATURE_COLUMNS, engineer_features

MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/Insightflow California Housing/production")

APP_NAME = "insightflow"
APP_VERSION = "1.0.0"


class PredictRequest(BaseModel):
    """Raw California Housing features. Values are in the dataset's units."""

    MedInc: float = Field(..., description="Median income in block group (tens of thousands).")
    HouseAge: float = Field(..., description="Median house age in block group.")
    AveRooms: float = Field(..., description="Average number of rooms per household.")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household.")
    Population: float = Field(..., description="Block group population.")
    AveOccup: float = Field(..., description="Average number of household members.")
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)


class PredictResponse(BaseModel):
    prediction: float
    units: str = "USD x100k"
    model_version: str | None = None


def load_model(model_uri: str = MODEL_URI):
    """Load the model from the MLflow registry (Production by default)."""
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as exc:  # model missing / unreachable registry
        raise RuntimeError(f"Failed to load model from '{model_uri}': {exc}") from exc


def create_app(model: Any | None = None) -> FastAPI:
    """Application factory. Tests may inject a model to avoid registry I/O."""
    app = FastAPI(title=APP_NAME, version=APP_VERSION)
    app.state.model = model if model is not None else load_model()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": APP_NAME}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        # Build a 1-row DataFrame in the exact training column order and
        # apply the same feature engineering used at training time.
        row = {col: [getattr(request, col)] for col in RAW_FEATURE_COLUMNS}
        import pandas as pd

        df = engineer_features(pd.DataFrame(row))
        if df.empty:
            raise HTTPException(status_code=422, detail="Features produced an empty row.")

        prediction = float(app.state.model.predict(df)[0])
        return PredictResponse(prediction=prediction, model_version=MODEL_URI)

    return app


load_dotenv()
app = create_app()
