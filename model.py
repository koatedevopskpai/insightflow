"""Model construction and training core.

`build_pipeline` and `train_and_evaluate` are framework-agnostic and
importable by both train.py (single runs) and optimize.py (Optuna sweep),
so the same code path is exercised in both.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_pipeline(n_estimators: int = 100, max_depth: int = 10) -> Pipeline:
    """Build the model as a deterministic sklearn pipeline.

    StandardScaler first so the RandomForest sees z-scored inputs; the
    scaler is fit only on training data inside Pipeline to avoid leakage.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> tuple[Pipeline, float]:
    """Train a pipeline and return (model, RMSE) on the held-out test set."""
    pipeline = build_pipeline(n_estimators, max_depth)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return pipeline, rmse
