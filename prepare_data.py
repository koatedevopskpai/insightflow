"""Prepare the California Housing dataset: load raw data, engineer features,
and split into versioned train/test CSVs.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from features import RAW_FEATURE_COLUMNS, TARGET_COLUMN, engineer_features


def main() -> None:
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=RAW_FEATURE_COLUMNS)
    df[TARGET_COLUMN] = housing.target

    # Feature engineering lives in features.py so training and serving
    # always apply the exact same transforms (no train/serve skew).
    df = engineer_features(df)

    # Deterministic split for reproducible experiments.
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print(f"Training set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    print("Columns after engineering:", list(df.columns))


if __name__ == "__main__":
    main()
