"""Dataset loading and splitting helpers.

Kept separate from model code so train.py and optimize.py share one
canonical way to turn the versioned CSVs into model-ready matrices.
"""

from pathlib import Path

import pandas as pd

from features import TARGET_COLUMN, feature_columns

TRAIN_CSV = Path("train.csv")
TEST_CSV = Path("test.csv")


def load_split(train_csv: str | Path = TRAIN_CSV, test_csv: str | Path = TEST_CSV):
    """Load the train/test CSVs and return (X_train, y_train, X_test, y_test)."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df[feature_columns(train_df)]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns(test_df)]
    y_test = test_df[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test
