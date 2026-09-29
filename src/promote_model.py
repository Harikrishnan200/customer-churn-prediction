"""
Promote a model version to an MLflow alias (e.g. "production").

WHAT: Points an alias (a movable label like "production" or
      "candidate") at a specific version of the registered model.
WHY:  The FastAPI app always loads whichever version holds the
      "production" alias. Promoting a new model is therefore just
      moving a label in MLflow — no code change, no redeploy of new
      model files needed.
WHERE: Run manually after training and evaluating a new model version,
       once you've decided it's good enough to serve.
HOW:  python -m src.promote_model --version 3 --alias production

This is intentionally a manual, human-triggered step (see
docs/learning-guide.md for why automatic promotion is out of scope).
"""

import argparse
import logging

import mlflow
import yaml
from mlflow import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Promote a model version to an alias.")
    parser.add_argument("--version", required=True, help="Model version number to promote, e.g. 3")
    parser.add_argument(
        "--alias",
        default="production",
        help="Alias to point at this version (default: production)",
    )
    args = parser.parse_args()

    config = load_config()
    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    client = MlflowClient()
    client.set_registered_model_alias(
        name=mlflow_cfg["model_name"],
        alias=args.alias,
        version=args.version,
    )
    logger.info(
        "Alias '%s' now points to %s version %s",
        args.alias,
        mlflow_cfg["model_name"],
        args.version,
    )


if __name__ == "__main__":
    main()
