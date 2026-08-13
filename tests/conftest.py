"""Shared fixtures for hermetic tests (no real data, no MLflow registry)."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """A small DataFrame with raw California Housing columns only."""
    n = 40
    rng = np.random.default_rng(42)
    data = {
        "MedInc": rng.uniform(1, 10, n),
        "HouseAge": rng.uniform(1, 50, n),
        "AveRooms": rng.uniform(3, 8, n),
        "AveBedrms": rng.uniform(0.5, 2, n),
        "Population": rng.uniform(100, 3000, n),
        "AveOccup": rng.uniform(1, 6, n),
        "Latitude": rng.uniform(32, 42, n),
        "Longitude": rng.uniform(-124, -114, n),
    }
    return pd.DataFrame(data)
