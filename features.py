"""Feature engineering shared by training and serving.

Centralizing feature transforms here prevents train/serve skew: the same
derived columns are guaranteed to be produced whether the DataFrame comes
from the raw dataset (prepare_data.py) or from a live API request
(serve.py).
"""

import numpy as np
import pandas as pd

# Raw feature columns present on the California Housing dataset.
RAW_FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Target column (median house value, in hundreds of thousands of dollars).
TARGET_COLUMN = "MedHouseVal"

# Derived features built on top of the raw columns.
DERIVED_FEATURE_COLUMNS = [
    "RoomsPerHousehold",
    "BedroomsPerRoom",
    "PopulationPerHousehold",
    "LogMedInc",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append derived features to ``df`` in a fixed column order.

    Derived features are computed from the raw columns; the raw columns are
    preserved so the feature matrix is the union of both groups. Column
    ordering is deterministic to keep the model's expected schema stable.
    """
    result = df.copy()

    # Derived ratios — all divide by occupancy or room counts, so guard
    # against division by zero by replacing infinities with NaN afterwards.
    result["RoomsPerHousehold"] = result["AveRooms"] / result["AveOccup"]
    result["BedroomsPerRoom"] = result["AveBedrms"] / result["AveRooms"]
    result["PopulationPerHousehold"] = result["Population"] / result["AveOccup"]

    # Log transform tames the heavily right-skewed median income feature.
    result["LogMedInc"] = np.log1p(result["MedInc"])

    # Remove any rows corrupted by division by zero.
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result.dropna(inplace=True)

    return result


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns used as model features, in stable order."""
    return [col for col in RAW_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS if col in df.columns]
