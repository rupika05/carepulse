"""
prediction_service.py — Model loading and prediction logic for the Healthcare Triage API.
"""
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    MODEL_PATH, CLASSES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    OUTPUTS_DIR, SUBMISSION_FILENAME, PATIENT_ID_COLUMN, TARGET_COLUMN
)
from src.cost_sensitive import predict_cost_sensitive_labels, get_expected_costs_dict, total_misclassification_cost


# Alias mapping for medical datasets and alternative schemas
COLUMN_ALIASES = {
    "pulse_rate": "heartrate",
    "pulse": "heartrate",
    "hr": "heartrate",
    "heart_rate": "heartrate",
    "respiratory_rate": "resprate",
    "resp_rate": "resprate",
    "respiration_rate": "resprate",
    "rr": "resprate",
    "systolic_bp": "sbp",
    "systolic": "sbp",
    "sbp_mmhg": "sbp",
    "diastolic_bp": "dbp",
    "diastolic": "dbp",
    "dbp_mmhg": "dbp",
    "spo2": "o2sat",
    "oxygen_saturation": "o2sat",
    "o2_sat": "o2sat",
    "sa_o2": "o2sat",
    "temperature_f": "temperature",
    "temp": "temperature",
    "gcs": "gcs_total",
    "glasgow_coma_scale": "gcs_total",
    "injury_type": "chiefcomplaint",
    "complaint": "chiefcomplaint",
    "reason_for_visit": "chiefcomplaint",
}

AVPU_MAP = {
    "Alert": "A", "alert": "A", "A": "A",
    "Voice": "V", "voice": "V", "V": "V",
    "Pain": "P", "pain": "P", "P": "P",
    "Unresponsive": "U", "unresponsive": "U", "U": "U",
}

AVPU_ORDINAL_MAP = {"A": 4.0, "V": 3.0, "P": 2.0, "U": 1.0}


def _synthesize_description(row: pd.Series) -> str:
    """Synthesize clinically realistic triage notes when description is absent."""
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

    if complaint and complaint.lower() not in ("nan", "none", ""):
        if transport and transport.lower() not in ("nan", "none", "unknown", ""):
            parts.append(f"Patient arrives via {transport} presenting with {complaint}.")
        else:
            parts.append(f"Patient presented with {complaint}.")
    else:
        parts.append("Patient presented for emergency triage evaluation.")

    if avpu == "U" or (pd.notna(rr) and rr == 0 and pd.notna(hr) and hr == 0):
        parts.append("Patient is unresponsive, flaccid, and unable to follow commands with severely depressed neurological status.")
    elif avpu == "P":
        parts.append("Patient exhibits altered mental status, responds only to painful stimuli, and does not follow verbal commands.")
    elif avpu == "V":
        parts.append("Patient is lethargic, responding intermittently to verbal stimuli with reduced awareness.")
    elif avpu == "A":
        if follows == 1 or follows == "1" or follows == 1.0:
            parts.append("Patient is fully alert, oriented, and promptly follows verbal commands.")
        else:
            parts.append("Patient is alert but displays mild confusion or agitation.")

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

    if bp_un == 1 or bp_un == "1" or bp_un == 1.0 or (pd.notna(sbp) and float(sbp) < 85):
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


def _enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, map aliases, derive missing vitals, and generate descriptions."""
    df = df.copy()

    # 1. Alias mapping
    for alias, target in COLUMN_ALIASES.items():
        if alias in df.columns and target not in df.columns:
            df[target] = df[alias]

    # 2. AVPU mapping
    if "avpu" in df.columns:
        df["avpu"] = df["avpu"].map(lambda x: AVPU_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)

    if "avpu_ordinal" not in df.columns and "avpu" in df.columns:
        df["avpu_ordinal"] = df["avpu"].map(AVPU_ORDINAL_MAP)

    if "follows_commands" not in df.columns and "avpu" in df.columns:
        df["follows_commands"] = df["avpu"].map(lambda x: 1.0 if x == "A" else 0.0)

    # 3. Derive missing vital sign aggregates if base vitals exist
    vital_pairs = [
        ("vs_heartrate", "heartrate"),
        ("vs_resprate", "resprate"),
        ("vs_sbp", "sbp"),
        ("vs_o2sat", "o2sat"),
    ]
    for prefix, base in vital_pairs:
        for suffix in ["_min", "_max", "_mean"]:
            col = f"{prefix}{suffix}"
            if col not in df.columns and base in df.columns:
                df[col] = df[base]

    if "vs_temperature_max" not in df.columns and "temperature" in df.columns:
        df["vs_temperature_max"] = df["temperature"]

    # 4. Handle blood pressure unobtainable flag
    if "bp_unobtainable" not in df.columns:
        if "sbp" in df.columns:
            df["bp_unobtainable"] = df["sbp"].map(lambda x: 1.0 if (pd.isna(x) or x == 0) else 0.0)
        else:
            df["bp_unobtainable"] = 0.0

    # 5. Generate clinical description if missing or empty
    has_desc = "description" in df.columns and df["description"].dropna().str.strip().ne("").any()
    if not has_desc:
        df["description"] = df.apply(_synthesize_description, axis=1)

    return df


class PredictionService:
    """Service that holds the loaded scikit-learn model pipeline."""

    def __init__(self):
        self._pipeline = None
        self._model_name: Optional[str] = None
        self._loaded: bool = False

    def load(self) -> None:
        """Load the model pipeline from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run `python -m src.train` from the project root first."
            )
        payload = joblib.load(MODEL_PATH)
        self._pipeline = payload["pipeline"]
        self._model_name = payload["model_name"]
        self._loaded = True
        print(f"[PredictionService] Model loaded: {self._model_name}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the prediction pipeline for one patient.
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Enrich single dictionary by wrapping in DataFrame
        df_single = pd.DataFrame([features])
        df_enriched = _enrich_dataframe(df_single)

        row = {}
        for col in NUMERICAL_FEATURES:
            val = df_enriched[col].iloc[0] if col in df_enriched.columns else None
            row[col] = [np.nan if (val is None or pd.isna(val)) else float(val)]

        for col in CATEGORICAL_FEATURES:
            val = df_enriched[col].iloc[0] if col in df_enriched.columns else None
            row[col] = [np.nan if (val is None or pd.isna(val)) else str(val)]

        desc = df_enriched["description"].iloc[0] if "description" in df_enriched.columns else ""
        row["description"] = [str(desc) if desc is not None else ""]

        df = pd.DataFrame(row)

        raw_proba = self._pipeline.predict_proba(df)

        pipeline_classes = list(self._pipeline.classes_)
        if pipeline_classes != CLASSES:
            reorder_idx = [pipeline_classes.index(cls) for cls in CLASSES]
            raw_proba = raw_proba[:, reorder_idx]

        proba_1d = raw_proba[0]

        # Cost-sensitive prediction
        triage_label = predict_cost_sensitive_labels(raw_proba)[0]

        # Expected costs per class
        expected_costs = get_expected_costs_dict(proba_1d)

        return {
            "triage": triage_label,
            "probabilities": {
                "RED": round(float(proba_1d[0]), 4),
                "YELLOW": round(float(proba_1d[1]), 4),
                "GREEN": round(float(proba_1d[2]), 4),
                "BLACK": round(float(proba_1d[3]), 4),
            },
            "expected_costs": expected_costs,
            "model_name": self._model_name,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run prediction on a batch DataFrame and return formatted results.
        Standardizes schemas, handles column aliases, synthesizes missing descriptions,
        and preserves actual outcomes.
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Check for actual target column before dropping/enriching
        actual_col = None
        for candidate in [
            TARGET_COLUMN, "actual_triage", "Actual Outcome", "Actual Triage",
            "triage_label", "actual", "Actual", "target", "Target"
        ]:
            if candidate in df.columns:
                actual_col = candidate
                break

        actual_values = df[actual_col].reset_index(drop=True) if actual_col else None

        # Keep patient IDs if present
        if PATIENT_ID_COLUMN in df.columns:
            patient_ids = df[PATIENT_ID_COLUMN].reset_index(drop=True)
        elif "Patient ID" in df.columns:
            patient_ids = df["Patient ID"].reset_index(drop=True)
        elif "patient_id" in df.columns:
            patient_ids = df["patient_id"].reset_index(drop=True)
        else:
            patient_ids = pd.Series([f"PAT_{i+1:04d}" for i in range(len(df))])

        # Enrich input DataFrame with aliases, derived features, and clinical narratives
        X = _enrich_dataframe(df)

        if actual_col and actual_col in X.columns:
            X = X.drop(columns=[actual_col])

        # Align columns
        for col in NUMERICAL_FEATURES:
            if col not in X.columns:
                X[col] = np.nan
        for col in CATEGORICAL_FEATURES:
            if col not in X.columns:
                X[col] = np.nan

        descriptions = X["description"].reset_index(drop=True)

        X = X[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["description"]]

        raw_proba = self._pipeline.predict_proba(X)
        pipeline_classes = list(self._pipeline.classes_)
        if pipeline_classes != CLASSES:
            reorder_idx = [pipeline_classes.index(cls) for cls in CLASSES]
            raw_proba = raw_proba[:, reorder_idx]

        cost_preds = predict_cost_sensitive_labels(raw_proba)

        result_dict = {
            "Patient ID": patient_ids,
            "Predicted Triage": cost_preds,
            "RED Probability": raw_proba[:, 0].round(6),
            "YELLOW Probability": raw_proba[:, 1].round(6),
            "GREEN Probability": raw_proba[:, 2].round(6),
            "BLACK Probability": raw_proba[:, 3].round(6),
            "Description": descriptions,
        }

        if actual_values is not None:
            result_dict["Actual Outcome"] = actual_values

        result_df = pd.DataFrame(result_dict)
        return result_df


# Module singleton
prediction_service = PredictionService()

