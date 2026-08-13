"""Freeze the Production model into a portable directory for serving.

The Model Registry's artifact URIs are machine-specific (absolute host
paths), so a container cannot resolve them from the SQLite DB alone.
This script downloads the Production model into models/production as a
plain MLflow model directory, which is then mounted into the serving
container (or baked into the image) — the same pattern teams use to
freeze immutable deployment artifacts.

Usage:
    python export_model.py [--model-uri models:/Insightflow California Housing/production]
"""

import argparse
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

DEFAULT_MODEL_URI = "models:/Insightflow California Housing/production"
EXPORT_DIR = Path("models/production")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a model from the registry.")
    parser.add_argument("--model-uri", default=os.getenv("MLFLOW_MODEL_URI", DEFAULT_MODEL_URI))
    parser.add_argument("--out", type=Path, default=EXPORT_DIR)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    args.out.mkdir(parents=True, exist_ok=True)
    model = mlflow.sklearn.load_model(args.model_uri)
    mlflow.sklearn.save_model(model, args.out)
    print(f"Exported model from '{args.model_uri}' to {args.out}")


if __name__ == "__main__":
    main(parse_args())
