"""
main.py — FastAPI backend for the Healthcare Triage Classification system (MM26ML03).
"""
import io
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.schemas import (
    PatientInput, PredictionResponse, HealthResponse
)
from backend.services.prediction_service import prediction_service
from src.config import (
    METRICS_PATH, OUTPUTS_DIR, SUBMISSION_FILENAME, TEST_PATH,
    CLASSES, COST_MATRIX, DATA_DIR, PROJECT_ROOT, TRAIN_PATH
)

from src.cost_sensitive import total_misclassification_cost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan: load model once at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        prediction_service.load()
        logger.info("Model loaded successfully at startup.")
    except Exception as e:
        logger.error(f"STARTUP ERROR: {e}")
    yield
    logger.info("Shutting down triage API.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Healthcare Triage Classification API",
    description="MM26ML03 — Emergency Triage Decision Support API.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns server health status and model load status."""
    return HealthResponse(
        status="ok",
        model_loaded=prediction_service.is_loaded,
        model_name=prediction_service.model_name,
        competition_id="MM26ML03",
    )


@app.get("/metrics", tags=["Analytics"])
async def get_metrics():
    """Returns cross-validation evaluation metrics and cost matrix configuration."""
    metrics_data = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
    
    return {
        "metrics": metrics_data,
        "classes": CLASSES,
        "cost_matrix": COST_MATRIX,
    }


ASSESSED_PATIENTS_PATH = DATA_DIR / "assessed_patients.csv"


def _append_to_assessed_dataset(patient_features: dict, result: dict, subject_id: str = None, patient_id: str = None) -> int:
    """Appends a newly assessed patient record and its prediction outcomes to assessed_patients.csv."""
    try:
        import datetime
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        pid = subject_id or patient_id or patient_features.get("patient_id") or patient_features.get("subject_id") or f"PAT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        record = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subject_id": str(pid),
        }

        # Core features
        feature_keys = [
            "gender", "race", "arrival_transport", "temperature", "heartrate", "resprate",
            "o2sat", "sbp", "dbp", "bp_unobtainable", "pain_score", "pain_assessable",
            "gcs_eye", "gcs_verbal", "gcs_motor", "gcs_total", "avpu", "avpu_ordinal",
            "follows_commands", "consciousness_source", "chiefcomplaint", "description",
            "n_vitalsign_readings", "n_diagnoses", "n_home_meds"
        ]
        for col in feature_keys:
            val = patient_features.get(col, None)
            record[col] = val if val is not None else ""

        # Prediction outcomes
        record["predicted_triage"] = result.get("triage", "")
        probs = result.get("probabilities", {})
        record["red_probability"] = probs.get("RED", "")
        record["yellow_probability"] = probs.get("YELLOW", "")
        record["green_probability"] = probs.get("GREEN", "")
        record["black_probability"] = probs.get("BLACK", "")

        row_df = pd.DataFrame([record])
        if not ASSESSED_PATIENTS_PATH.exists():
            row_df.to_csv(ASSESSED_PATIENTS_PATH, index=False)
        else:
            row_df.to_csv(ASSESSED_PATIENTS_PATH, mode="a", header=False, index=False)

        total_count = len(pd.read_csv(ASSESSED_PATIENTS_PATH))
        logger.info(f"Saved assessed patient {pid} ({record['predicted_triage']}) to {ASSESSED_PATIENTS_PATH}. Total: {total_count}")
        return total_count
    except Exception as e:
        logger.error(f"Failed to record assessed patient to CSV: {e}", exc_info=True)
        return -1


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Predict triage category for a single patient with cost-sensitive decision theory
    and automatically append the patient record + prediction outcome to the assessed_patients.csv dataset.
    """
    if not prediction_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please ensure the model file is generated."
        )

    features = patient.model_dump()
    subject_id = features.pop("subject_id", None)
    patient_id = features.pop("patient_id", None)

    try:
        result = prediction_service.predict(features)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Append to assessed_patients.csv dataset
    total_records = _append_to_assessed_dataset(features, result, subject_id, patient_id)

    return PredictionResponse(
        triage=result["triage"],
        probabilities=result["probabilities"],
        expected_costs=result["expected_costs"],
        model_name=result["model_name"],
        subject_id=subject_id,
        patient_id=subject_id or patient_id,
        saved_to_csv=True,
        csv_file="data/assessed_patients.csv",
        total_assessed_records=total_records,
    )


@app.get("/download-assessed-patients", tags=["Export"])
async def download_assessed_patients():
    """Download the CSV dataset containing all recorded single patient assessments and prediction outcomes."""
    if not ASSESSED_PATIENTS_PATH.exists():
        # If no assessments yet, create an empty CSV template
        headers = [
            "timestamp", "subject_id", "gender", "race", "arrival_transport", "temperature",
            "heartrate", "resprate", "o2sat", "sbp", "dbp", "bp_unobtainable", "pain_score",
            "pain_assessable", "gcs_eye", "gcs_verbal", "gcs_motor", "gcs_total", "avpu",
            "avpu_ordinal", "follows_commands", "consciousness_source", "chiefcomplaint",
            "description", "n_vitalsign_readings", "n_diagnoses", "n_home_meds",
            "predicted_triage", "red_probability", "yellow_probability", "green_probability", "black_probability"
        ]
        ASSESSED_PATIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=headers).to_csv(ASSESSED_PATIENTS_PATH, index=False)

    return FileResponse(
        path=str(ASSESSED_PATIENTS_PATH),
        filename="assessed_patients.csv",
        media_type="text/csv",
    )


@app.get("/assessed-patients", tags=["Analytics"])
async def get_assessed_patients():
    """Return summary and recent records from assessed_patients.csv."""
    if not ASSESSED_PATIENTS_PATH.exists():
        return {"total": 0, "patients": []}
    try:
        df = pd.read_csv(ASSESSED_PATIENTS_PATH).fillna("")
        records = df.tail(50).to_dict(orient="records")
        return {"total": len(df), "patients": records}
    except Exception:
        return {"total": 0, "patients": []}




def _build_batch_response(pred_df: pd.DataFrame, filename: str, column_warnings=None):
    # Predicted distribution across all classes
    pred_dist = {cls: 0 for cls in CLASSES}
    for cls, count in pred_df["Predicted Triage"].value_counts().items():
        if cls in pred_dist:
            pred_dist[cls] = int(count)

    has_actuals = "Actual Outcome" in pred_df.columns
    actual_dist = {cls: 0 for cls in CLASSES}
    accuracy = None
    total_cost = None
    confusion_dict = None

    if has_actuals:
        # Standardize strings
        pred_df["Actual Outcome"] = pred_df["Actual Outcome"].astype(str).str.strip().str.upper()
        valid_mask = pred_df["Actual Outcome"].isin(CLASSES)
        
        if valid_mask.any():
            for cls, count in pred_df.loc[valid_mask, "Actual Outcome"].value_counts().items():
                if cls in actual_dist:
                    actual_dist[cls] = int(count)

            correct = (pred_df.loc[valid_mask, "Actual Outcome"] == pred_df.loc[valid_mask, "Predicted Triage"]).sum()
            accuracy = round(float(correct / valid_mask.sum()), 4)
            
            try:
                total_cost = float(total_misclassification_cost(
                    pred_df.loc[valid_mask, "Actual Outcome"].values,
                    pred_df.loc[valid_mask, "Predicted Triage"].values
                ))
            except Exception:
                total_cost = None

            # Compute confusion matrix
            confusion_dict = {}
            for actual_cls in CLASSES:
                confusion_dict[actual_cls] = {}
                for pred_cls in CLASSES:
                    match_count = int(((pred_df["Actual Outcome"] == actual_cls) & (pred_df["Predicted Triage"] == pred_cls)).sum())
                    confusion_dict[actual_cls][pred_cls] = match_count

    items = []
    has_desc = "Description" in pred_df.columns
    for _, row in pred_df.iterrows():
        is_match = bool(row["Predicted Triage"] == row["Actual Outcome"]) if has_actuals and pd.notna(row["Actual Outcome"]) and row["Actual Outcome"] in CLASSES else None
        items.append({
            "subject_id": str(row["Patient ID"]),
            "description": str(row["Description"]) if has_desc and pd.notna(row["Description"]) else None,
            "actual_outcome": str(row["Actual Outcome"]) if has_actuals and pd.notna(row["Actual Outcome"]) and row["Actual Outcome"] in CLASSES else None,
            "predicted_triage": str(row["Predicted Triage"]),
            "is_correct": is_match,
            "red_probability": float(row["RED Probability"]),
            "yellow_probability": float(row["YELLOW Probability"]),
            "green_probability": float(row["GREEN Probability"]),
            "black_probability": float(row["BLACK Probability"]),
        })

    return {
        "filename": filename,
        "total_patients": len(pred_df),
        "distribution": pred_dist,
        "actual_distribution": actual_dist if has_actuals else None,
        "has_actuals": has_actuals,
        "accuracy": accuracy,
        "total_cost": total_cost,
        "confusion_matrix": confusion_dict,
        "column_warnings": column_warnings if column_warnings else None,
        "predictions": items,
    }


@app.post("/batch-predict", tags=["Inference"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Upload a CSV file containing patient records to receive batch predictions,
    triage urgency classifications, actual vs predicted comparison, and distribution metrics.
    """
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    pred_df = prediction_service.predict_batch(df)
    return _build_batch_response(pred_df, filename=file.filename)


@app.get("/test-predictions", tags=["Inference"])
async def get_test_predictions():
    """Run batch prediction on the official competition test dataset (test_pred_ml03.csv)."""
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    test_path = TEST_PATH if TEST_PATH.exists() else PROJECT_ROOT / "test_pred_ml03.csv"
    if not test_path.exists():
        raise HTTPException(status_code=404, detail="Test dataset file not found.")

    test_df = pd.read_csv(test_path)
    pred_df = prediction_service.predict_batch(test_df)
    return _build_batch_response(pred_df, filename=test_path.name)


@app.get("/load-sample", tags=["Inference"])
async def load_sample_dataset(name: str = "ground_truth"):
    """
    Quickly load and evaluate a benchmark or test dataset.
    Supported names: 'ground_truth' (750 patients with actual labels), 'competition_test' (112 patients).
    """
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if name in ("ground_truth", "test_ground_truth"):
        gt_path = DATA_DIR / "test_ground_truth.csv"
        if not gt_path.exists():
            gt_path = PROJECT_ROOT / "data" / "test_ground_truth.csv"
        if not gt_path.exists():
            raise HTTPException(status_code=404, detail="Ground truth benchmark file not found.")
        df = pd.read_csv(gt_path)
        pred_df = prediction_service.predict_batch(df)
        return _build_batch_response(pred_df, filename="test_ground_truth.csv (Benchmark)")

    elif name in ("competition_test", "test_pred_ml03", "test"):
        test_path = TEST_PATH if TEST_PATH.exists() else PROJECT_ROOT / "test_pred_ml03.csv"
        if not test_path.exists():
            raise HTTPException(status_code=404, detail="Competition test file not found.")
        df = pd.read_csv(test_path)
        pred_df = prediction_service.predict_batch(df)
        return _build_batch_response(pred_df, filename="test_pred_ml03.csv (Competition Test)")

    elif name in ("train", "train_ml03"):
        train_path = TRAIN_PATH if TRAIN_PATH.exists() else PROJECT_ROOT / "train_ml03.csv"
        if not train_path.exists():
            raise HTTPException(status_code=404, detail="Training file not found.")
        df = pd.read_csv(train_path)
        pred_df = prediction_service.predict_batch(df)
        return _build_batch_response(pred_df, filename="train_ml03.csv (Training Cohort)")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown sample dataset: {name}. Use 'ground_truth', 'competition_test', or 'train'.")



@app.post("/check-predictions", tags=["Verification"])
async def check_predictions(
    predictions_file: UploadFile = File(...),
    ground_truth_file: UploadFile = File(...),
):
    """
    Compare predicted triage labels against ground-truth labels.
    Accepts two CSV files:
      - predictions_file: must contain 'subject_id'/'Patient ID' and 'Predicted Triage'/'predicted_triage' columns.
      - ground_truth_file: must contain an ID column and a triage label column (e.g., start_category, triage_label).
    Returns per-row match status, accuracy, macro-F1, weighted-F1, total cost, and confusion matrix.
    """
    try:
        pred_contents = await predictions_file.read()
        pred_df = pd.read_csv(io.BytesIO(pred_contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid predictions CSV: {str(e)}")

    try:
        truth_contents = await ground_truth_file.read()
        truth_df = pd.read_csv(io.BytesIO(truth_contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ground truth CSV: {str(e)}")

    # --- Find the prediction column ---
    pred_col = None
    for candidate in ["Predicted Triage", "predicted_triage", "prediction", "Prediction", "predicted", "Predicted"]:
        if candidate in pred_df.columns:
            pred_col = candidate
            break
    if pred_col is None:
        if prediction_service.is_loaded:
            batch_res = prediction_service.predict_batch(pred_df)
            pred_df["Predicted Triage"] = batch_res["Predicted Triage"]
            pred_col = "Predicted Triage"
            for prob_c in ["RED Probability", "YELLOW Probability", "GREEN Probability", "BLACK Probability"]:
                if prob_c in batch_res.columns:
                    pred_df[prob_c] = batch_res[prob_c]
        else:
            raise HTTPException(status_code=400, detail="Predictions CSV must have a 'Predicted Triage' or 'predicted_triage' column.")

    # --- Find the prediction ID column ---
    pred_id_col = None
    for candidate in ["Patient ID", "subject_id", "patient_id", "PatientID", "id", "ID"]:
        if candidate in pred_df.columns:
            pred_id_col = candidate
            break
    if pred_id_col is None:
        # Fallback: use index-based matching
        pred_df["_match_idx"] = range(len(pred_df))
        pred_id_col = "_match_idx"

    # --- Find the ground truth label column ---
    truth_label_col = None
    for candidate in ["start_category", "triage_label", "Actual Outcome", "actual_triage", "Actual", "actual", "target", "Target", "label", "Label"]:
        if candidate in truth_df.columns:
            truth_label_col = candidate
            break
    if truth_label_col is None:
        raise HTTPException(status_code=400, detail=f"Ground truth CSV must have a triage label column (e.g., start_category, triage_label). Found columns: {list(truth_df.columns)}")

    # --- Find the ground truth ID column ---
    truth_id_col = None
    for candidate in ["subject_id", "Patient ID", "patient_id", "PatientID", "id", "ID"]:
        if candidate in truth_df.columns:
            truth_id_col = candidate
            break

    # Merge or align
    if truth_id_col and pred_id_col != "_match_idx":
        # Merge on ID
        pred_df[pred_id_col] = pred_df[pred_id_col].astype(str).str.strip()
        truth_df[truth_id_col] = truth_df[truth_id_col].astype(str).str.strip()
        merged = pred_df.merge(truth_df[[truth_id_col, truth_label_col]], left_on=pred_id_col, right_on=truth_id_col, how="inner")
        if len(merged) == 0:
            # Try index-based matching as fallback
            min_len = min(len(pred_df), len(truth_df))
            merged = pred_df.head(min_len).copy()
            merged["_truth"] = truth_df[truth_label_col].head(min_len).values
            truth_label_col = "_truth"
    else:
        # Index-based matching
        min_len = min(len(pred_df), len(truth_df))
        merged = pred_df.head(min_len).copy()
        merged["_truth"] = truth_df[truth_label_col].head(min_len).values
        truth_label_col = "_truth"

    # Standardize labels
    merged[pred_col] = merged[pred_col].astype(str).str.strip().str.upper()
    merged[truth_label_col] = merged[truth_label_col].astype(str).str.strip().str.upper()

    # Filter to valid classes only
    valid_mask = merged[truth_label_col].isin(CLASSES) & merged[pred_col].isin(CLASSES)
    valid_df = merged[valid_mask].copy()

    if len(valid_df) == 0:
        raise HTTPException(status_code=400, detail="No valid triage labels found after merging. Ensure both files use RED/YELLOW/GREEN/BLACK labels.")

    y_true = valid_df[truth_label_col].tolist()
    y_pred = valid_df[pred_col].tolist()

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
    macro_f1 = round(float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)), 4)
    weighted_f1 = round(float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)), 4)

    try:
        cost = float(total_misclassification_cost(y_true, y_pred))
    except Exception:
        cost = None

    # Confusion matrix
    confusion_dict = {}
    for actual_cls in CLASSES:
        confusion_dict[actual_cls] = {}
        for p_cls in CLASSES:
            confusion_dict[actual_cls][p_cls] = int(sum(1 for a, p in zip(y_true, y_pred) if a == actual_cls and p == p_cls))

    # Per-row results
    items = []
    id_col_for_output = pred_id_col if pred_id_col != "_match_idx" else None
    for _, row in valid_df.iterrows():
        is_correct = row[pred_col] == row[truth_label_col]
        item = {
            "subject_id": str(row[id_col_for_output]) if id_col_for_output else str(row.name),
            "predicted_triage": row[pred_col],
            "actual_triage": row[truth_label_col],
            "is_correct": bool(is_correct),
        }
        # Include probability columns if present
        for prob_col, key in [("RED Probability", "red_probability"), ("YELLOW Probability", "yellow_probability"),
                               ("GREEN Probability", "green_probability"), ("BLACK Probability", "black_probability"),
                               ("red_probability", "red_probability"), ("yellow_probability", "yellow_probability"),
                               ("green_probability", "green_probability"), ("black_probability", "black_probability")]:
            if prob_col in row.index and pd.notna(row[prob_col]):
                item[key] = round(float(row[prob_col]), 6)
        items.append(item)

    # Distributions
    pred_dist = {cls: int(sum(1 for p in y_pred if p == cls)) for cls in CLASSES}
    actual_dist = {cls: int(sum(1 for a in y_true if a == cls)) for cls in CLASSES}

    correct_count = sum(1 for a, p in zip(y_true, y_pred) if a == p)
    incorrect_count = len(y_true) - correct_count

    return {
        "total_checked": len(valid_df),
        "total_in_predictions": len(pred_df),
        "total_in_ground_truth": len(truth_df),
        "matched_rows": len(valid_df),
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_cost": cost,
        "predicted_distribution": pred_dist,
        "actual_distribution": actual_dist,
        "confusion_matrix": confusion_dict,
        "results": items,
    }


@app.get("/download-submission", tags=["Export"])
async def download_submission():
    """Download the competition submission CSV file."""
    submission_path = OUTPUTS_DIR / SUBMISSION_FILENAME
    if not submission_path.exists():
        test_df = pd.read_csv(TEST_PATH)
        pred_df = prediction_service.predict_batch(test_df)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(submission_path, index=False)

    return FileResponse(
        path=str(submission_path),
        filename=SUBMISSION_FILENAME,
        media_type="text/csv",
    )


@app.get("/word-importances", tags=["Analytics"])
async def get_word_importances():
    """Return the top predictive words and clinical phrases extracted by Random Forest."""
    word_file = OUTPUTS_DIR / "rf_word_importances.json"
    if not word_file.exists():
        return {"words": []}
    with open(word_file, "r") as f:
        data = json.load(f)
    return {"words": data}

