"""
generate_dataset.py — Generates a clinically-grounded synthetic dataset
based on the START (Simple Triage and Rapid Treatment) triage protocol.

This script is used ONCE to create train.csv and test.csv.
The data distributions are derived from published START triage decision rules,
not invented arbitrarily.

START Triage Decision Rules:
  BLACK  → No respirations (after repositioning), or respirations present but
            GCS ≤ 8 and respiration rate > 30 or < 6 with absent radial pulse
  RED    → Respirations present AND (RR > 30, or radial pulse absent,
            or GCS < 13, or SpO2 < 90)
  YELLOW → Ambulatory-NO but does not meet RED/BLACK criteria
  GREEN  → Ambulatory (walking wounded), minor/no injuries

Reference: START triage algorithm (SALT/START mass casualty framework).
This dataset is SYNTHETIC and intended for educational/hackathon use only.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import (
    DATA_DIR, TRAIN_PATH, TEST_PATH, RANDOM_SEED,
    TARGET_COLUMN, PATIENT_ID_COLUMN
)

RNG = np.random.default_rng(RANDOM_SEED)


def _generate_patients(n: int, start_id: int = 0) -> pd.DataFrame:
    """Generate n synthetic patients with START-protocol-consistent labels."""
    records = []
    pid = start_id

    while len(records) < n:
        pid += 1

        # --- Demographic ---
        age = int(RNG.integers(1, 96))
        gender = RNG.choice(["M", "F"])

        # --- Start by sampling a triage class with realistic MCI frequencies ---
        # In mass casualty events: ~30% RED, ~35% YELLOW, ~25% GREEN, ~10% BLACK
        triage_class = RNG.choice(
            ["RED", "YELLOW", "GREEN", "BLACK"],
            p=[0.28, 0.35, 0.27, 0.10]
        )

        # --- Generate vitals consistent with triage class ---
        if triage_class == "BLACK":
            rr = RNG.choice(
                [0, RNG.integers(35, 50)],
                p=[0.6, 0.4]
            )
            rr = int(rr) if hasattr(rr, 'item') else int(rr)
            pulse = int(RNG.choice([0, RNG.integers(120, 180)], p=[0.5, 0.5]))
            sbp = int(RNG.integers(0, 60)) if pulse == 0 else int(RNG.integers(40, 80))
            spo2 = int(RNG.integers(50, 82))
            gcs = int(RNG.integers(3, 7))
            avpu = "Unresponsive"
            resp_effort = RNG.choice(["Absent", "Labored"], p=[0.65, 0.35])
            radial_pulse = "No"
            cap_refill = round(float(RNG.uniform(4.0, 6.0)), 1)
            major_hemorrhage = RNG.choice(["Yes", "No"], p=[0.4, 0.6])
            ambulatory = "No"
            injury_type = RNG.choice(["Blunt", "Penetrating", "Burns"], p=[0.4, 0.4, 0.2])
            injury_severity = RNG.choice(["Severe", "Critical"], p=[0.3, 0.7])
            mechanism = RNG.choice(["MVA", "Explosion", "GSW", "Fall"], p=[0.3, 0.3, 0.2, 0.2])

        elif triage_class == "RED":
            rr = int(RNG.choice(
                [RNG.integers(31, 40), RNG.integers(1, 5)],
                p=[0.7, 0.3]
            ))
            pulse = int(RNG.integers(100, 150))
            sbp = int(RNG.integers(70, 100))
            spo2 = int(RNG.integers(78, 90))
            gcs = int(RNG.integers(8, 13))
            avpu = RNG.choice(["Pain", "Voice"], p=[0.6, 0.4])
            resp_effort = RNG.choice(["Labored", "Normal"], p=[0.75, 0.25])
            radial_pulse = RNG.choice(["No", "Yes"], p=[0.6, 0.4])
            cap_refill = round(float(RNG.uniform(2.5, 4.5)), 1)
            major_hemorrhage = RNG.choice(["Yes", "No"], p=[0.55, 0.45])
            ambulatory = "No"
            injury_type = RNG.choice(["Blunt", "Penetrating", "Burns"], p=[0.5, 0.35, 0.15])
            injury_severity = RNG.choice(["Moderate", "Severe", "Critical"], p=[0.2, 0.5, 0.3])
            mechanism = RNG.choice(["MVA", "Explosion", "GSW", "Stab", "Fall"], p=[0.3, 0.2, 0.2, 0.15, 0.15])

        elif triage_class == "YELLOW":
            rr = int(RNG.integers(12, 30))
            pulse = int(RNG.integers(60, 110))
            sbp = int(RNG.integers(90, 130))
            spo2 = int(RNG.integers(90, 96))
            gcs = int(RNG.integers(13, 15))
            avpu = RNG.choice(["Alert", "Voice"], p=[0.6, 0.4])
            resp_effort = RNG.choice(["Normal", "Labored"], p=[0.8, 0.2])
            radial_pulse = "Yes"
            cap_refill = round(float(RNG.uniform(1.5, 3.0)), 1)
            major_hemorrhage = RNG.choice(["Yes", "No"], p=[0.2, 0.8])
            ambulatory = "No"
            injury_type = RNG.choice(["Blunt", "Penetrating", "Burns", "None"], p=[0.4, 0.2, 0.1, 0.3])
            injury_severity = RNG.choice(["Minor", "Moderate"], p=[0.4, 0.6])
            mechanism = RNG.choice(["MVA", "Fall", "Stab", "Other"], p=[0.35, 0.35, 0.15, 0.15])

        else:  # GREEN
            rr = int(RNG.integers(12, 25))
            pulse = int(RNG.integers(60, 100))
            sbp = int(RNG.integers(100, 140))
            spo2 = int(RNG.integers(95, 100))
            gcs = 15
            avpu = "Alert"
            resp_effort = "Normal"
            radial_pulse = "Yes"
            cap_refill = round(float(RNG.uniform(0.5, 2.0)), 1)
            major_hemorrhage = "No"
            ambulatory = "Yes"
            injury_type = RNG.choice(["Blunt", "None", "Burns"], p=[0.4, 0.45, 0.15])
            injury_severity = RNG.choice(["None", "Minor"], p=[0.5, 0.5])
            mechanism = RNG.choice(["Fall", "MVA", "Other", "None"], p=[0.35, 0.25, 0.2, 0.2])

        # --- Introduce realistic missing values (~5% rate) ---
        def maybe_nan(value, p_missing=0.04):
            return np.nan if RNG.random() < p_missing else value

        records.append({
            PATIENT_ID_COLUMN: f"PAT_{pid:05d}",
            "age": maybe_nan(age),
            "gender": gender,
            "respiratory_rate": maybe_nan(rr),
            "pulse_rate": maybe_nan(pulse),
            "systolic_bp": maybe_nan(sbp),
            "spo2": maybe_nan(spo2),
            "gcs_total": maybe_nan(gcs),
            "avpu": avpu,
            "respiratory_effort": resp_effort,
            "radial_pulse_present": radial_pulse,
            "capillary_refill_sec": maybe_nan(cap_refill),
            "major_hemorrhage": major_hemorrhage,
            "ambulatory": ambulatory,
            "injury_type": injury_type,
            "injury_severity": injury_severity,
            "mechanism_of_injury": mechanism,
            TARGET_COLUMN: triage_class,
        })

    return pd.DataFrame(records[:n])


def generate_and_save(n_train: int = 3000, n_test: int = 750) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_train} training patients...")
    train_df = _generate_patients(n_train, start_id=0)

    print(f"Generating {n_test} test patients...")
    test_df = _generate_patients(n_test, start_id=n_train)
    # Test set: keep patient_id, drop target label (as in real competition)
    test_df_no_label = test_df.drop(columns=[TARGET_COLUMN])

    # Save ground truth for test (used in evaluation scripts only)
    test_df.to_csv(DATA_DIR / "test_ground_truth.csv", index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df_no_label.to_csv(TEST_PATH, index=False)

    print("\n=== Dataset Report ===")
    print(f"Train: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    print(f"Test:  {test_df_no_label.shape[0]} rows × {test_df_no_label.shape[1]} columns")
    print(f"\nTrain columns: {list(train_df.columns)}")
    print(f"\nTarget distribution (train):")
    dist = train_df[TARGET_COLUMN].value_counts()
    for cls, cnt in dist.items():
        print(f"  {cls}: {cnt} ({cnt/len(train_df)*100:.1f}%)")
    print(f"\nMissing values (train):")
    missing = train_df.isnull().sum()
    for col, cnt in missing[missing > 0].items():
        print(f"  {col}: {cnt} ({cnt/len(train_df)*100:.1f}%)")
    print(f"\nSaved: {TRAIN_PATH}")
    print(f"Saved: {TEST_PATH}")
    print(f"Saved: {DATA_DIR / 'test_ground_truth.csv'} (for evaluation only)")


if __name__ == "__main__":
    generate_and_save()
