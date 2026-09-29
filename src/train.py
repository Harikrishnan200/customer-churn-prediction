"""
Training script for ChurnOps.

WHAT: Trains a churn classifier and logs everything about the training
      run (parameters, metrics, artifacts, the model itself) to MLflow.
WHY:  Without experiment tracking, every training run is a black box —
      you can't compare "today's model" to "last week's model", or
      remember which hyperparameters produced which results. MLflow
      solves that by recording every run automatically.
WHERE: This is the only place a model gets trained in this project.
HOW:  Run `python -m src.train` after `python -m src.data`. Make sure an
      MLflow server is running first (see README) so the run has
      somewhere to log to.

MLflow concepts used here, explained once so they aren't a mystery:
- Experiment:  a named folder that groups related runs together
               (here: "churnops"). All churn-model training runs live
               inside this one experiment.
- Run:         one execution of training. Each time you call
               `python -m src.train`, a new run is created with its own
               parameters, metrics and artifacts.
- Parameter:   a training input you chose, e.g. model_type or
               random_state. Logged once at the start of a run.
- Metric:      a number that measures how well the model performed,
               e.g. accuracy or f1. Logged after evaluation.
- Artifact:    any file produced by the run that's worth keeping, e.g.
               a confusion matrix image or the serialized model itself.
- Registered model: a named, version-controlled entry in the MLflow
               Model Registry (here: "churn-model"). Every time we
               register a new model, it gets a new version number
               (v1, v2, v3, ...) without overwriting previous ones.
- Alias:       a movable pointer to one specific model version, e.g.
               "production" or "candidate". The API always loads
               whichever version currently holds the "production"
               alias, so promoting a new model is just moving the
               alias — no code change needed.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # no GUI needed; we only save plots to disk

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = ["contract", "internet_service"]
# senior_citizen, partner, dependents are already 0/1 integers - no
# preprocessing needed, so they are passed straight through.
PASSTHROUGH_FEATURES = ["senior_citizen", "partner", "dependents"]
TARGET = "churn"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_pipeline(model_type: str, random_state: int) -> Pipeline:
    """
    Build one sklearn Pipeline that does preprocessing AND prediction.

    WHY a single pipeline instead of preprocessing the data separately:
    if preprocessing lived outside the model (e.g. duplicated inside
    FastAPI), it would be very easy for training-time and serving-time
    preprocessing to drift apart — a classic, hard-to-debug production
    bug. Bundling everything into one Pipeline object means MLflow logs
    (and the API later loads) ONE artifact that always preprocesses
    input exactly the way it was trained to.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("passthrough", "passthrough", PASSTHROUGH_FEATURES),
        ]
    )

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Compute the standard classification metrics used to judge the model.

    - accuracy: how many predictions were correct overall. Can be
      misleading on imbalanced data (e.g. if only 20% of customers
      churn, predicting "no churn" for everyone gives 80% accuracy
      while being useless).
    - precision: of the customers we PREDICTED would churn, how many
      actually did? High precision = few wasted retention offers.
    - recall: of the customers who ACTUALLY churned, how many did we
      catch? High recall = few missed at-risk customers.
    - f1: harmonic mean of precision and recall - a single number that
      balances both.
    - roc_auc: how well the model ranks churners above non-churners
      across all possible thresholds, independent of any one cutoff.

    There is no single "best" metric here — it depends on the business
    cost of a false positive (wasted retention budget) versus a false
    negative (a churned customer we didn't try to retain).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def save_confusion_matrix(y_true, y_pred, output_path: str) -> None:
    """Save a confusion matrix plot as a PNG so it can be logged to MLflow."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    config = load_config()
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    # --- Load processed data (created by `python -m src.data`) ---
    processed_dir = config["data"]["processed_dir"]
    train_df = pd.read_csv(f"{processed_dir}/train.csv")
    val_df = pd.read_csv(f"{processed_dir}/val.csv")

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
    X_train, y_train = train_df[feature_columns], train_df[TARGET]
    X_val, y_val = val_df[feature_columns], val_df[TARGET]

    # --- Point MLflow at the tracking server and experiment ---
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run():
        logger.info("Started MLflow run")

        # Parameters: the inputs that define how this run was configured.
        # Logging them lets us reproduce or compare runs later.
        mlflow.log_param("model_type", train_cfg["model_type"])
        mlflow.log_param("test_size", train_cfg["test_size"])
        mlflow.log_param("random_state", train_cfg["random_state"])
        mlflow.log_param("train_rows", len(X_train))

        pipeline = build_pipeline(train_cfg["model_type"], train_cfg["random_state"])
        pipeline.fit(X_train, y_train)
        logger.info("Model trained")

        # Evaluate on the validation set - data the model never trained on.
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_pred, y_proba)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        logger.info("Validation metrics: %s", metrics)

        # Artifacts: files worth keeping alongside the run for inspection.
        save_confusion_matrix(y_val, y_pred, "reports/confusion_matrix.png")
        mlflow.log_artifact("reports/confusion_matrix.png")

        report_text = classification_report(y_val, y_pred, target_names=["No Churn", "Churn"])
        with open("reports/classification_report.txt", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("reports/classification_report.txt")

        # Log the trained pipeline (preprocessing + model together) and
        # register it as a new version of "churn-model" in the Model
        # Registry in one step.
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=mlflow_cfg["model_name"],
            input_example=X_train.head(2),
        )
        logger.info("Model logged and registered as '%s'", mlflow_cfg["model_name"])

    logger.info("Training run complete. View it at %s", mlflow_cfg["tracking_uri"])


if __name__ == "__main__":
    main()
