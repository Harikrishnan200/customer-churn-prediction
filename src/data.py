"""
Data pipeline for ChurnOps.

WHAT: This module turns the raw IBM Telco Customer Churn CSV into clean,
      split, ready-to-train datasets.
WHY:  Raw data almost always has messy values (blank strings, inconsistent
      types). Models need clean, numeric-friendly, well-typed data. Keeping
      this logic in one place means training and future retraining always
      produce data in the same shape.
WHERE: Used by src/train.py (to build the training set) and
       src/monitor.py (to build the reference/current datasets for drift
       detection).
HOW:  Run directly with `python -m src.data` to regenerate
      data/processed/{train,val,test,reference}.csv from the raw CSV.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Only these raw columns are used. The rest of the Telco dataset (gender,
# phone service details, etc.) is dropped on purpose: the goal of this
# project is to demonstrate MLOps, not to squeeze out extra ML accuracy by
# using every available column. This subset also matches exactly the
# fields the FastAPI /predict endpoint accepts.
RAW_COLUMNS = {
    "tenure": "tenure",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Contract": "contract",
    "InternetService": "internet_service",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "Churn": "churn",
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the single YAML config file used across the whole project."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(raw_path: str) -> pd.DataFrame:
    """Read the raw CSV from disk exactly as downloaded, no changes."""
    logger.info("Loading raw data from %s", raw_path)
    return pd.read_csv(raw_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the columns we care about and fix known data-quality issues.

    Known issue in this dataset: TotalCharges is stored as text and has
    blank values (" ") for customers with tenure == 0 (brand new
    customers who haven't been billed yet). We convert those to 0.0
    rather than dropping the rows, since a new customer is a valid,
    common case the model should still be able to score.
    """
    df = df[list(RAW_COLUMNS.keys())].rename(columns=RAW_COLUMNS)

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(0.0)

    # Yes/No text columns -> 0/1 integers, matching the API's request schema.
    df["partner"] = (df["partner"] == "Yes").astype(int)
    df["dependents"] = (df["dependents"] == "Yes").astype(int)

    # This is the prediction target: 1 = customer churned, 0 = stayed.
    df["churn"] = (df["churn"] == "Yes").astype(int)

    return df


def split_data(df: pd.DataFrame, test_size: float, val_size: float, random_state: int):
    """
    Split into train / validation / test sets.

    WHY three sets and not just two:
    - train: what the model learns from.
    - validation: used during development to check the model isn't
      overfitting, before we ever look at the test set.
    - test: touched only once, at the end, to report an honest, unbiased
      estimate of real-world performance.
    """
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["churn"]
    )
    # val_size is expressed as a fraction of the *original* dataset, so we
    # rescale it to be a fraction of the remaining train_val_df.
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df["churn"],
    )
    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df, processed_dir: str) -> None:
    """
    Write the split datasets to disk as CSVs.

    We also save train.csv a second time as reference.csv: this becomes
    the "reference" (expected/normal) distribution that src/monitor.py
    compares future production data against to detect drift.
    """
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    train_df.to_csv(out_dir / "reference.csv", index=False)

    logger.info(
        "Saved processed data: train=%d val=%d test=%d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )


def main():
    config = load_config()

    df = load_data(config["data"]["raw_path"])
    df = clean_data(df)

    train_df, val_df, test_df = split_data(
        df,
        test_size=config["training"]["test_size"],
        val_size=config["training"]["val_size"],
        random_state=config["training"]["random_state"],
    )

    save_processed_data(train_df, val_df, test_df, config["data"]["processed_dir"])


if __name__ == "__main__":
    main()
