"""Tests for the FastAPI serving layer (hermetic: injected model, no registry)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from serve import create_app


@pytest.fixture
def dummy_model():
    """A real sklearn pipeline that accepts the 12 engineered features."""
    rng = np.random.default_rng(0)
    col_names = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
        "AveOccup", "Latitude", "Longitude", "RoomsPerHousehold",
        "BedroomsPerRoom", "PopulationPerHousehold", "LogMedInc",
    ]
    X = pd.DataFrame(rng.normal(size=(200, 12)), columns=col_names)
    y = rng.normal(size=200)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def client(dummy_model):
    app = create_app(model=dummy_model)
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_number(client):
    payload = {
        "MedInc": 8.3,
        "HouseAge": 30,
        "AveRooms": 6.1,
        "AveBedrms": 1.1,
        "Population": 1000,
        "AveOccup": 3.2,
        "Latitude": 37.7,
        "Longitude": -122.2,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["prediction"], float)
    assert np.isfinite(body["prediction"])


def test_predict_rejects_missing_field(client):
    payload = {
        "MedInc": 8.3,
        "HouseAge": 30,
        "AveRooms": 6.1,
        "AveBedrms": 1.1,
        "Population": 1000,
        "AveOccup": 3.2,
        "Latitude": 37.7,
        # Longitude missing
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_latitude(client):
    payload = {
        "MedInc": 8.3,
        "HouseAge": 30,
        "AveRooms": 6.1,
        "AveBedrms": 1.1,
        "Population": 1000,
        "AveOccup": 3.2,
        "Latitude": 999,
        "Longitude": -122.2,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
