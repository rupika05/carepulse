"""
predict.py — Inference script for generating competition submission.

Loads the saved model pipeline, generates predictions on the test set,
applies the cost-sensitive decision layer, and outputs the required CSV.

Usage:
    python -m src.predict
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    MODEL_PATH, TEST_PATH, OUTPUTS_DIR, CLASSES,
    PATIENT_ID_COLUMN, TARGET_COLUMN, TEAM_ID, SUBMISSION_FILENAME
)
from src.data_processing import load_test
from src.cost_sensitive import predict_cost_sensitive_labels, compute_expected_costs_batch


def load_pipeline():
    """Load the saved model pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    payload = joblib.load(MODEL_PATH)
    print(f"[OK] Loaded model: {payload['model_name']}")
    return payload["pipeline"], payload["model_name"]


def generate_predictions(
    pipeline,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate class probabilities and cost-sensitive predictions for test set.
    """
    # Keep patient IDs for output
    patient_ids = test_df[PATIENT_ID_COLUMN].reset_index(drop=True)

    # Drop patient_id before inference
    X_test = test_df.drop(columns=[PATIENT_ID_COLUMN], errors="ignore").copy()

    # Ensure description column exists
    if "description" not in X_test.columns:
        if "chiefcomplaint" in X_test.columns:
            X_test["description"] = X_test["chiefcomplaint"].fillna("").astype(str)
        else:
            X_test["description"] = ""

    # Get probabilities (n_samples, 4) — order matches pipeline.classes_
    raw_proba = pipeline.predict_proba(X_test)

    # Ensure column order matches CLASSES = [RED, YELLOW, GREEN, BLACK]
    pipeline_classes = list(pipeline.classes_)
    if pipeline_classes != CLASSES:
        # Reorder columns to match canonical class order
        reorder_idx = [pipeline_classes.index(cls) for cls in CLASSES]
        raw_proba = raw_proba[:, reorder_idx]

    # Cost-sensitive predictions
    cost_predictions = predict_cost_sensitive_labels(raw_proba)

    results = pd.DataFrame({
        "Patient ID": patient_ids,
        "Predicted Triage": cost_predictions,
        "RED Probability": raw_proba[:, 0].round(6),
        "YELLOW Probability": raw_proba[:, 1].round(6),
        "GREEN Probability": raw_proba[:, 2].round(6),
        "BLACK Probability": raw_proba[:, 3].round(6),
    })
    return results


def validate_submission(submission_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validate the submission CSV before saving."""
    errors = []

    # Row count
    if len(submission_df) != len(test_df):
        errors.append(f"Row count mismatch: submission={len(submission_df)}, test={len(test_df)}")

    # Patient ID match
    if not submission_df["Patient ID"].equals(test_df[PATIENT_ID_COLUMN]):
        errors.append("Patient IDs do not match test data.")

    # Valid labels
    valid_labels = set(CLASSES)
    invalid = set(submission_df["Predicted Triage"]) - valid_labels
    if invalid:
        errors.append(f"Invalid triage labels found: {invalid}")

    # Probability range
    prob_cols = ["RED Probability", "YELLOW Probability", "GREEN Probability", "BLACK Probability"]
    for col in prob_cols:
        if not ((submission_df[col] >= 0) & (submission_df[col] <= 1)).all():
            errors.append(f"Probabilities out of [0, 1] range in column {col}")

    # Probabilities sum to ~1
    prob_sums = submission_df[prob_cols].sum(axis=1)
    if not ((prob_sums - 1.0).abs() < 0.01).all():
        errors.append("Some probability rows do not sum to approximately 1.")

    if errors:
        raise ValueError("Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    print(f"[OK] Submission validation passed ({len(submission_df)} rows)")


def save_submission(submission_df: pd.DataFrame) -> Path:
    """Save the submission CSV in the required format."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / SUBMISSION_FILENAME
    submission_df.to_csv(path, index=False)
    print(f"[OK] Submission saved to {path}")
    return path


def main():
    print("=" * 60)
    print("  Healthcare Triage -- Generating Test Predictions")
    print("=" * 60)

    pipeline, model_name = load_pipeline()
    test_df = load_test()
    print(f"Test set: {len(test_df)} patients")

    submission_df = generate_predictions(pipeline, test_df)

    print("\nPrediction distribution:")
    dist = submission_df["Predicted Triage"].value_counts()
    for cls in CLASSES:
        cnt = dist.get(cls, 0)
        print(f"  {cls:<8} {cnt:>4} ({cnt/len(submission_df)*100:.1f}%)")

    validate_submission(submission_df, test_df)

    path = save_submission(submission_df)

    # Also save to standard predictions path
    submission_df.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    print("\n=== Submission Summary ===")
    print(f"File:      {path.name}")
    print(f"Rows:      {len(submission_df)}")
    print(submission_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
