"""
Model loading and prediction logic for the ChurnOps API.

WHAT: Loads the "production" version of the churn model from the MLflow
      Model Registry and wraps it with a simple predict() function.
WHY:  Loading a model from MLflow involves a network call and
      deserializing a scikit-learn pipeline - relatively slow. Doing
      that on every request would add unnecessary latency and load on
      the MLflow server. Instead we load the model exactly once, when
      the FastAPI app starts, and keep it in memory for the life of the
      process.
WHERE: Used by app/main.py at startup (load_model) and in the
       /predict endpoint (predict).
"""

import logging
import os

import mlflow
import pandas as pd
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

# By default, the MLflow client retries a slow/unreachable server for
# several minutes before giving up, which would make the API hang on
# startup if MLflow isn't running yet. We fail fast instead - a
# developer or CI job should see "model failed to load" within a few
# seconds, not after a multi-minute stall. setdefault() means a real
# deployment can still override these via the environment if needed.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

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


class ChurnPredictor:
    """Holds one loaded model in memory and serves predictions from it."""

    def __init__(self, tracking_uri: str, model_name: str, model_alias: str):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model_alias = model_alias
        self.model = None
        self.model_version = None

    def load(self) -> None:
        """
        Load the model version currently behind `model_alias` (normally
        "production") from the MLflow Model Registry.

        Raises whatever MLflow raises if the server is unreachable or the
        alias doesn't exist yet - app/main.py turns that into a clear
        startup failure rather than a silent, broken predictor.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        model_uri = f"models:/{self.model_name}@{self.model_alias}"

        logger.info("Loading model from %s", model_uri)
        self.model = mlflow.sklearn.load_model(model_uri)

        client = MlflowClient()
        version_info = client.get_model_version_by_alias(self.model_name, self.model_alias)
        self.model_version = version_info.version

        logger.info(
            "Loaded model '%s' version %s (alias '%s')",
            self.model_name,
            self.model_version,
            self.model_alias,
        )

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, request: dict) -> tuple[bool, float]:
        """
        Turn one validated request dict into a churn prediction.

        The input dict already matches the column names the training
        pipeline expects, so we just wrap it in a single-row DataFrame -
        the pipeline itself handles scaling/encoding exactly as it did
        during training.
        """
        input_df = pd.DataFrame([request], columns=FEATURE_COLUMNS)

        probability = float(self.model.predict_proba(input_df)[0][1])
        churn = probability >= 0.5

        return churn, probability
