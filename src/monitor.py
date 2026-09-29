"""
Data drift detection for ChurnOps, using Evidently.

WHAT: Compares a "reference" dataset (what training data normally looks
      like) against a "current" dataset (a stand-in for recent
      production traffic) and reports whether feature distributions
      have shifted.
WHY:  A model is trained on a snapshot of the world. If the real world
      changes - customers' typical tenure shifts, a new contract type
      becomes common, pricing changes - the model's assumptions become
      stale even though nothing about the model itself changed. This is
      called data drift, and it's one of the most common causes of a
      model silently getting worse in production over time.
WHERE: Run manually/periodically, independent of the training pipeline.
HOW:  python -m src.monitor
      Produces an HTML report at reports/drift_report.html that you can
      open in a browser.

IMPORTANT: this script only detects and reports drift - it does NOT
retrain the model automatically. See docs/learning-guide.md for why
automatic retraining is intentionally out of scope for this project:
in short, a human should look at *why* the data changed before
deciding whether retraining (or something else entirely, like a data
pipeline bug) is the right response.
"""

import logging

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_drift_report(reference_path: str, current_path: str, report_path: str) -> None:
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    # DataDriftPreset compares the statistical distribution of every
    # shared column between the two datasets (e.g. via a
    # Kolmogorov-Smirnov test for numeric columns) and flags any column
    # whose distribution has shifted beyond a threshold.
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(report_path)

    logger.info("Drift report saved to %s", report_path)

    result = report.as_dict()
    dataset_drift = result["metrics"][0]["result"]["dataset_drift"]
    n_drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]
    logger.info("Dataset drift detected: %s (%d drifted column(s))", dataset_drift, n_drifted)


def main():
    config = load_config()
    monitoring_cfg = config["monitoring"]

    run_drift_report(
        reference_path=monitoring_cfg["reference_data"],
        current_path=monitoring_cfg["current_data"],
        report_path=monitoring_cfg["report_path"],
    )


if __name__ == "__main__":
    main()
