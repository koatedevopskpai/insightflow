"""Bayesian hyperparameter optimization with Optuna, plus model promotion.

Runs a study over the training function in model.py, logs every trial to
MLflow, then registers the best candidate in the Model Registry and
promotes it to Production — but only if it beats the incumbent.

The promotion gate is the ML equivalent of refusing a merge that fails CI:
a candidate never silently replaces a deployed model.

Usage:
    python optimize.py --n-trials 20 --min-rmse 0.9
"""

import argparse
import os

import mlflow
import optuna
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from data import load_split
from model import train_and_evaluate

EXPERIMENT_NAME = "California Housing Optuna"
MODEL_NAME = "Insightflow California Housing"
SAMPLER_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the California Housing model.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--min-rmse",
        type=float,
        default=0.75,
        help="Quality gate: refuse to promote candidates above this RMSE.",
    )
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    return parser.parse_args()


def objective(
    trial: optuna.Trial,
    X_train,
    y_train,
    X_test,
    y_test,
) -> float:
    """Single Optuna trial: suggest hyperparameters, train, log, return RMSE."""
    n_estimators = trial.suggest_int("n_estimators", 50, 300, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 20)

    with mlflow.start_run():
        mlflow.log_params({"trial_number": trial.number})
        pipeline, rmse = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        mlflow.log_params(
            {"n_estimators": n_estimators, "max_depth": max_depth}
        )
        mlflow.log_metric("rmse", rmse)

        # Every trial is deployable, so log its model artifact too.
        mlflow.sklearn.log_model(pipeline, "model")

    return rmse


def best_run(experiment_name: str):
    """Return the lowest-RMSE run from the given experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["metrics.rmse ASC"]
    )
    return runs[0] if runs else None


def current_production_rmse(model_name: str) -> float | None:
    """RMSE of the incumbent Production model, if one exists."""
    client = MlflowClient()
    try:
        latest = client.get_latest_versions(model_name, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        return None
    if not latest:
        return None
    run_id = latest[0].run_id
    run = client.get_run(run_id)
    return run.data.metrics.get("rmse")


def promote(best: dict, model_name: str, min_rmse: float) -> None:
    """Register the best run's model and promote it with a quality gate."""
    client = MlflowClient()

    incumbent = current_production_rmse(model_name)
    candidate_rmse = best.data.metrics["rmse"]
    if candidate_rmse > min_rmse:
        raise SystemExit(
            f"Candidate RMSE {candidate_rmse:.4f} exceeds gate {min_rmse:.4f}; "
            "refusing to promote."
        )
    if incumbent is not None and candidate_rmse >= incumbent:
        raise SystemExit(
            f"Candidate RMSE {candidate_rmse:.4f} not better than incumbent "
            f"{incumbent:.4f}; refusing to promote."
        )

    model_uri = f"runs:/{best.info.run_id}/model"
    registered = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(model_name, registered.version, "Staging")
    client.transition_model_version_stage(model_name, registered.version, "Production")
    client.update_registered_model(
        model_name,
        description=f"Promoted from Optuna sweep; RMSE={candidate_rmse:.4f}.",
    )
    print(
        f"Promoted model version {registered.version} to Production "
        f"(RMSE={candidate_rmse:.4f})."
    )


def main(args: argparse.Namespace) -> None:
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train, X_test, y_test = load_split()

    # Bayesian search: TPE builds a probabilistic model of good regions of
    # the search space and proposes candidates accordingly.
    sampler = optuna.samplers.TPESampler(seed=SAMPLER_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test),
        n_trials=args.n_trials,
    )

    print(f"Best trial: RMSE={study.best_value:.4f} params={study.best_params}")
    best = best_run(args.experiment)
    promote(best, MODEL_NAME, args.min_rmse)


if __name__ == "__main__":
    main(parse_args())
