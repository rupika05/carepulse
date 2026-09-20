"""
add_descriptions.py — Enrich dataset with realistic clinical descriptions/notes.

Generates clinical narrative summaries based on patient presentation,
chief complaints, vital signs, neurological status, and pain scores.
Does NOT leak the target class (RED/YELLOW/GREEN/BLACK).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")


def generate_clinical_description(row: pd.Series) -> str:
    """Synthesize a clinically realistic triage note from patient row."""
    parts = []
    transport = str(row.get("arrival_transport", "")).strip()
    complaint = str(row.get("chiefcomplaint", "")).strip()
    avpu = str(row.get("avpu", "")).strip()
    hr = row.get("heartrate")
    rr = row.get("resprate")
    sbp = row.get("sbp")
    o2 = row.get("o2sat")
    bp_un = row.get("bp_unobtainable")
    pain = row.get("pain_score")
    follows = row.get("follows_commands")

    # 1. Chief Complaint & Arrival Mode
    if complaint and complaint.lower() != "nan":
        if transport and transport.lower() != "nan":
            parts.append(f"Patient arrives via {transport} presenting with {complaint}.")
        else:
            parts.append(f"Patient presented with {complaint}.")
    else:
        parts.append("Patient presented for emergency triage evaluation.")

    # 2. Neurological & Consciousness
    if avpu == "U" or (pd.notna(rr) and rr == 0 and pd.notna(hr) and hr == 0):
        parts.append("Patient is unresponsive, flaccid, and unable to follow commands with severely depressed neurological status.")
    elif avpu == "P":
        parts.append("Patient exhibits altered mental status, responds only to painful stimuli, and does not follow verbal commands.")
    elif avpu == "V":
        parts.append("Patient is lethargic, responding intermittently to verbal stimuli with reduced awareness.")
    elif avpu == "A":
        if follows == 1 or follows == "1":
            parts.append("Patient is fully alert, oriented, and promptly follows verbal commands.")
        else:
            parts.append("Patient is alert but displays mild confusion or agitation.")

    # 3. Respiratory Status
    if pd.notna(rr):
        try:
            r_val = float(rr)
            if r_val == 0:
                parts.append("Patient is apneic with absent spontaneous respirations, requiring immediate airway management.")
            elif r_val > 28:
                parts.append(f"Severe tachypnea noted with rapid labored breathing at {int(r_val)} breaths/min and respiratory distress.")
            elif r_val < 10:
                parts.append(f"Bradypnea and shallow respiratory depression noted at {int(r_val)} breaths/min.")
            else:
                parts.append(f"Spontaneous respirations are regular and unlabored at {int(r_val)} breaths/min.")
        except (ValueError, TypeError):
            pass

    # 4. Hemodynamics & Oxygenation
    hemo = []
    if pd.notna(hr):
        try:
            h_val = float(hr)
            if h_val == 0:
                hemo.append("absent peripheral pulse, pulseless electrical activity or cardiac arrest")
            elif h_val > 115:
                hemo.append(f"marked tachycardia with heart rate {int(h_val)} bpm")
            elif h_val < 50:
                hemo.append(f"marked bradycardia with heart rate {int(h_val)} bpm")
            else:
                hemo.append(f"stable pulse rate of {int(h_val)} bpm")
        except (ValueError, TypeError):
            pass

    if bp_un == 1 or bp_un == "1" or (pd.notna(sbp) and float(sbp) < 85):
        hemo.append("hypotensive shock with unobtainable or profoundly low blood pressure")
    elif pd.notna(sbp):
        try:
            hemo.append(f"systolic blood pressure measured at {int(float(sbp))} mmHg")
        except (ValueError, TypeError):
            pass

    if pd.notna(o2):
        try:
            o_val = float(o2)
            if o_val < 90:
                hemo.append(f"hypoxia with pulse oximetry of {int(o_val)}% on room air")
            else:
                hemo.append(f"adequate oxygen saturation at {int(o_val)}%")
        except (ValueError, TypeError):
            pass

    if hemo:
        parts.append("Vitals reveal " + ", ".join(hemo) + ".")

    # 5. Pain & Distress
    if pd.notna(pain):
        try:
            p_val = float(pain)
            if p_val >= 7:
                parts.append(f"Patient reports severe acute pain rated at {int(p_val)}/10.")
            elif p_val >= 4:
                parts.append(f"Moderate localized distress with pain score {int(p_val)}/10.")
            elif p_val > 0:
                parts.append(f"Mild discomfort with pain score {int(p_val)}/10.")
        except (ValueError, TypeError):
            pass

    return " ".join(parts)


def process_dataset(filepath: Path) -> pd.DataFrame:
    print(f">> Processing: {filepath}")
    df = pd.read_csv(filepath)
    df["description"] = df.apply(generate_clinical_description, axis=1)
    df.to_csv(filepath, index=False)
    print(f"   [OK] Saved {len(df)} rows with 'description' column to {filepath}")
    return df


def main():
    print("=" * 60)
    print("  Enriching Datasets with Clinical Descriptions")
    print("=" * 60)

    # 1. Train dataset
    train_ml03 = DATA_DIR / "train_ml03.csv"
    train_csv = DATA_DIR / "train.csv"

    if train_ml03.exists():
        df_train = process_dataset(train_ml03)
        # Also copy/save to train.csv
        df_train.to_csv(train_csv, index=False)
        print(f"   [OK] Synced to {train_csv}")
    elif train_csv.exists():
        process_dataset(train_csv)

    # 2. Test datasets
    test_csv = DATA_DIR / "test.csv"
    if test_csv.exists():
        process_dataset(test_csv)

    test_pred_ml03 = DATA_DIR / "test_pred_ml03.csv"
    if test_pred_ml03.exists():
        process_dataset(test_pred_ml03)

    print("\nSample Generated Description:")
    sample_df = pd.read_csv(train_csv)
    for i in range(min(3, len(sample_df))):
        print(f"--- Patient {sample_df.iloc[i]['subject_id']} ({sample_df.iloc[i].get('start_category', 'N/A')}) ---")
        print(sample_df.iloc[i]["description"])
        print()


if __name__ == "__main__":
    main()
