"""Tests for feature engineering (the train/serve contract)."""

import numpy as np
import pandas as pd

from features import (
    DERIVED_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    engineer_features,
    feature_columns,
)


def test_engineer_adds_all_derived_columns(sample_raw_df):
    result = engineer_features(sample_raw_df)
    assert all(col in result.columns for col in DERIVED_FEATURE_COLUMNS)
    # Raw columns are preserved.
    assert all(col in result.columns for col in RAW_FEATURE_COLUMNS)


def test_engineer_keeps_raw_rows_when_no_bad_divisions(sample_raw_df):
    result = engineer_features(sample_raw_df)
    assert len(result) == len(sample_raw_df)


def test_engineer_handles_zero_occupancy():
    # AveOccup == 0 would cause division by zero; rows must be dropped, not crash.
    df = pd.DataFrame(
        {
            "MedInc": [5.0, 5.0],
            "HouseAge": [20, 20],
            "AveRooms": [5, 5],
            "AveBedrms": [1, 1],
            "Population": [500, 500],
            "AveOccup": [0, 3],
            "Latitude": [37.7, 37.7],
            "Longitude": [-122.2, -122.2],
        }
    )
    result = engineer_features(df)
    assert len(result) == 1
    assert result["PopulationPerHousehold"].iloc[0] == 500 / 3


def test_engineer_log_transform_of_income(sample_raw_df):
    result = engineer_features(sample_raw_df)
    expected = np.log1p(sample_raw_df["MedInc"])
    np.testing.assert_allclose(result["LogMedInc"], expected)


def test_engineer_output_has_no_inf_or_nan(sample_raw_df):
    result = engineer_features(sample_raw_df)
    assert result.select_dtypes(include=[np.number]).isna().sum().sum() == 0
    assert np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all()


def test_feature_columns_order_is_stable(sample_raw_df):
    result = engineer_features(sample_raw_df)
    cols = feature_columns(result)
    # Raw columns come first, derived after — matches training expectations.
    assert cols == RAW_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS
