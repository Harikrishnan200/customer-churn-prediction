"""Tests for the data pipeline (src/data.py)."""

import pandas as pd

from src.data import clean_data, load_data, split_data


def test_load_data_reads_raw_csv():
    df = load_data("data/raw/churn.csv")
    assert len(df) > 0
    assert "Churn" in df.columns


def test_clean_data_produces_expected_columns():
    raw_df = load_data("data/raw/churn.csv")
    clean_df = clean_data(raw_df)

    expected_columns = {
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract",
        "internet_service",
        "senior_citizen",
        "partner",
        "dependents",
        "churn",
    }
    assert expected_columns == set(clean_df.columns)


def test_clean_data_target_is_binary():
    raw_df = load_data("data/raw/churn.csv")
    clean_df = clean_data(raw_df)

    assert set(clean_df["churn"].unique()) <= {0, 1}


def test_clean_data_has_no_missing_total_charges():
    raw_df = load_data("data/raw/churn.csv")
    clean_df = clean_data(raw_df)

    assert clean_df["total_charges"].isna().sum() == 0


def test_split_data_sizes_and_no_overlap():
    df = pd.DataFrame(
        {
            "tenure": range(100),
            "churn": [0, 1] * 50,
        }
    )
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1, random_state=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
