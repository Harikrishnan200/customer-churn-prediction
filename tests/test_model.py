"""
Tests for the training pipeline (src/train.py).

These tests build and fit the pipeline directly on a small in-memory
dataset. They deliberately do NOT talk to MLflow - MLflow tracking is
about recording runs, not about whether the model itself works, and
requiring a running MLflow server would make these tests slow and
environment-dependent.
"""

import pandas as pd

from src.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PASSTHROUGH_FEATURES,
    build_pipeline,
    compute_metrics,
)


def _toy_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": [1, 24, 60, 5, 36, 2, 48, 10],
            "monthly_charges": [70.0, 50.0, 20.0, 90.0, 40.0, 85.0, 30.0, 95.0],
            "total_charges": [70.0, 1200.0, 1200.0, 450.0, 1440.0, 170.0, 1440.0, 950.0],
            "contract": [
                "Month-to-month",
                "Two year",
                "Two year",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
            ],
            "internet_service": [
                "Fiber optic",
                "DSL",
                "No",
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "No",
                "Fiber optic",
            ],
            "senior_citizen": [0, 0, 1, 0, 0, 1, 0, 0],
            "partner": [0, 1, 1, 0, 1, 0, 1, 0],
            "dependents": [0, 1, 0, 0, 1, 0, 0, 0],
            "churn": [1, 0, 0, 1, 0, 1, 0, 1],
        }
    )


def test_build_pipeline_trains_and_predicts():
    df = _toy_dataset()
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
    X, y = df[feature_columns], df["churn"]

    pipeline = build_pipeline(model_type="logistic_regression", random_state=42)
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions) <= {0, 1}


def test_pipeline_predict_proba_returns_valid_probabilities():
    df = _toy_dataset()
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
    X, y = df[feature_columns], df["churn"]

    pipeline = build_pipeline(model_type="logistic_regression", random_state=42)
    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)[:, 1]
    assert len(probabilities) == len(X)
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_compute_metrics_returns_all_expected_keys():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_proba = [0.1, 0.9, 0.4, 0.2, 0.8]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
