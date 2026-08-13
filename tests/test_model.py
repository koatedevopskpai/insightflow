"""Tests for the training core (model.py)."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline

from model import build_pipeline, train_and_evaluate


@pytest.fixture
def tiny_split():
    """Small engineered feature matrix + targets (no external data files)."""
    X, y = make_regression(
        n_samples=300,
        n_features=len(
            ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
             "AveOccup", "Latitude", "Longitude", "RoomsPerHousehold",
             "BedroomsPerRoom", "PopulationPerHousehold", "LogMedInc"]
        ),
        n_informative=12,
        noise=0.1,
        random_state=42,
    )
    split = 240
    return X[:split], y[:split], X[split:], y[split:]


def test_build_pipeline_returns_pipeline():
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    assert "scaler" in pipe.named_steps
    assert "regressor" in pipe.named_steps


def test_train_and_evaluate_returns_positive_rmse(tiny_split):
    X_train, y_train, X_test, y_test = tiny_split
    _, rmse = train_and_evaluate(X_train, y_train, X_test, y_test, n_estimators=10, max_depth=5)
    assert rmse > 0
    assert np.isfinite(rmse)


def test_deeper_models_overfit_in_sample(tiny_split):
    """Sanity check: training RMSE improves with capacity (guards the metrics wiring)."""
    X_train, y_train, X_test, y_test = tiny_split
    _, rmse_small = train_and_evaluate(X_train, y_train, X_test, y_test, n_estimators=10, max_depth=2)
    _, rmse_large = train_and_evaluate(X_train, y_train, X_test, y_test, n_estimators=10, max_depth=10)
    # Deeper tree should fit the training data at least as well.
    assert rmse_large <= rmse_small
