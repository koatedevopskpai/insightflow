"""Train a single California Housing model run, logged to MLflow.

The tracking backend, experiment, and hyperparameters are all configurable
via CLI flags or environment variables, so the same script drives both
local experiments and a remote MLflow tracking server without edits.

Usage:
    python train.py --n-estimators 100 --max-depth 10
"""

import argparse
import os

import mlflow
from dotenv import load_dotenv

from data import load_split
from model import train_and_evaluate

EXPERIMENT_NAME = "California Housing Baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a California Housing model.")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)

    X_train, y_train, X_test, y_test = load_split()

    # Autolog captures sklearn params, metrics, and the model artifact.
    mlflow.autolog(log_models=True)

    with mlflow.start_run():
        _, rmse = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        mlflow.log_params(
            {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
        )
        mlflow.log_metric("rmse", rmse)

        print(
            f"Run logged: n_estimators={args.n_estimators}, "
            f"max_depth={args.max_depth}, RMSE={rmse:.4f}"
        )


if __name__ == "__main__":
    main(parse_args())
