"""
data_processing.py — Dataset loading, inspection, and validation utilities.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from src.config import (
    TRAIN_PATH, TEST_PATH, TARGET_COLUMN, PATIENT_ID_COLUMN,
    CLASSES, ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)


def load_train() -> pd.DataFrame:
    """Load training data from disk."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAIN_PATH}. "
                                 "Run generate_dataset.py first.")
    return pd.read_csv(TRAIN_PATH)


def load_test() -> pd.DataFrame:
    """Load test data from disk (no target column)."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found at {TEST_PATH}. "
                                 "Run generate_dataset.py first.")
    return pd.read_csv(TEST_PATH)


def inspect_dataset(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    Print a concise inspection report and return a summary dict.
    Does NOT modify the dataframe.
    """
    report: Dict[str, Any] = {}

    print(f"\n{'='*60}")
    print(f"  {name} Inspection Report")
    print(f"{'='*60}")

    # Shape
    report["rows"] = df.shape[0]
    report["columns"] = df.shape[1]
    print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

    # Column names and types
    print("\nColumn Types:")
    type_summary: Dict[str, str] = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique(dropna=False)
        print(f"  {col:<30} {dtype:<12} {n_unique:>6} unique")
        type_summary[col] = dtype
    report["dtypes"] = type_summary

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["missing"] > 0]
    print(f"\nMissing Values: {missing.sum()} total")
    if len(missing_df) > 0:
        for col, row in missing_df.iterrows():
            print(f"  {col:<30} {int(row['missing']):>5} ({row['pct']:.1f}%)")
    else:
        print("  None")
    report["missing"] = missing_df.to_dict()

    # Duplicate rows
    n_dups = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_dups}")
    report["duplicates"] = int(n_dups)

    # Target distribution (if present)
    if TARGET_COLUMN in df.columns:
        print(f"\nTarget Distribution ({TARGET_COLUMN}):")
        dist = df[TARGET_COLUMN].value_counts()
        for cls in CLASSES:
            cnt = dist.get(cls, 0)
            pct = cnt / len(df) * 100
            bar = "#" * int(pct / 2)
            print(f"  {cls:<8} {cnt:>5} ({pct:5.1f}%)  {bar}")
        report["class_distribution"] = dist.to_dict()

    # Column classification
    print(f"\nColumn Classification:")
    print(f"  Identifier: {[PATIENT_ID_COLUMN]}")
    print(f"  Numerical:  {NUMERICAL_FEATURES}")
    print(f"  Categorical:{CATEGORICAL_FEATURES}")
    if TARGET_COLUMN in df.columns:
        print(f"  Target:     [{TARGET_COLUMN}]")

    # Basic stats for numerical features
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if num_cols:
        print(f"\nNumerical Feature Stats:")
        stats = df[num_cols].describe().round(2)
        print(stats.to_string())
        report["numerical_stats"] = stats.to_dict()

    print(f"{'='*60}\n")
    return report


def validate_column_match(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Ensure train and test have the same feature columns (excluding target)."""
    train_features = set(train_df.columns) - {TARGET_COLUMN, PATIENT_ID_COLUMN}
    test_features = set(test_df.columns) - {PATIENT_ID_COLUMN}
    if train_features != test_features:
        only_train = train_features - test_features
        only_test = test_features - train_features
        raise ValueError(
            f"Column mismatch between train and test.\n"
            f"  Only in train: {only_train}\n"
            f"  Only in test:  {only_test}"
        )
    print("[OK] Train and test feature columns match.")


def check_data_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Check for patient ID overlap between train and test."""
    train_ids = set(train_df[PATIENT_ID_COLUMN])
    test_ids = set(test_df[PATIENT_ID_COLUMN])
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} patient IDs appear in both train and test!")
    print(f"[OK] No patient ID overlap between train and test. No leakage detected.")


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) dropping patient_id and target."""
    X = df.drop(columns=[TARGET_COLUMN, PATIENT_ID_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN]
    return X, y


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return X (features only) dropping patient_id."""
    return df.drop(columns=[PATIENT_ID_COLUMN], errors="ignore")
