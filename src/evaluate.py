"""
Standalone evaluation script for ChurnOps.

WHAT: Loads a model version from the MLflow Model Registry and evaluates
      it on the held-out test set (data the model has never seen, not
      even during validation).
WHY:  src/train.py already evaluates on the validation set during
      training. This script answers a different question: "how does a
      specific registered model version perform on completely unseen
      data?" — the same check you would run before promoting a
      candidate model to production.
WHERE: Run manually, after registering a model, before promoting it.
HOW:  `python -m src.evaluate` evaluates whichever version currently
      holds the "candidate" alias (or pass --alias production).
"""

import argparse
import logging

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
    "internet_service",
    "senior_citizen",
    "partner",
    "dependents",
]
TARGET = "churn"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a registered model on the test set.")
    parser.add_argument(
        "--alias",
        default="candidate",
        help="Model alias to evaluate, e.g. 'candidate' or 'production' (default: candidate)",
    )
    args = parser.parse_args()

    config = load_config()
    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    model_uri = f"models:/{mlflow_cfg['model_name']}@{args.alias}"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    test_df = pd.read_csv(f"{config['data']['processed_dir']}/test.csv")
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    logger.info("Test set metrics for alias '%s':", args.alias)
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)


if __name__ == "__main__":
    main()
