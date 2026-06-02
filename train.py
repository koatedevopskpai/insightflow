# train.py
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Set the MLflow tracking URI to a local directory (or can use a remote server)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("California Housing Baseline")

# Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameters to test
n_estimators = 100
max_depth = 5

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Build pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,  # noqa : E501
                ),
            ),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Log metric
    mlflow.log_metric("rmse", rmse)

    # Log the model
    mlflow.sklearn.log_model(pipeline, "model")

    print(
        f"Run logged: n_estimators={n_estimators}, max_depth={max_depth}, RMSE={rmse:.4f}"  # noqa : E501
    )
