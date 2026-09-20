"""
config.py — Central configuration for the Healthcare Triage Classification project.
All constants, paths, real dataset columns, and the competition cost matrix live here.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv" if (DATA_DIR / "train.csv").exists() else DATA_DIR / "train_ml03.csv"
TEST_PATH = DATA_DIR / "test.csv"

# ── Artifact paths ────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "final_model.joblib"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Real Dataset Column Definitions ───────────────────────────────────────────
TARGET_COLUMN = "start_category"
PATIENT_ID_COLUMN = "subject_id"

# Canonical class order (cost matrix depends on this order)
CLASSES = ["RED", "YELLOW", "GREEN", "BLACK"]
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}

# ── Competition cost matrix ───────────────────────────────────────────────────
# Rows = Actual class, Columns = Predicted class
# Order: RED=0, YELLOW=1, GREEN=2, BLACK=3
#
#               Predicted
# Actual     RED  YELLOW  GREEN  BLACK
# RED          0       5     10      5
# YELLOW       2       0      3      2
# GREEN        1       1      0      1
# BLACK        2       2      2      0
#
# Source: Competition problem statement MM26ML03. Do NOT modify.
COST_MATRIX = [
    [0, 5, 10, 5],   # actual = RED
    [2, 0,  3, 2],   # actual = YELLOW
    [1, 1,  0, 1],   # actual = GREEN
    [2, 2,  2, 0],   # actual = BLACK
]

# ── Cross-validation ──────────────────────────────────────────────────────────
CV_FOLDS = 5

# ── Submission filename format ────────────────────────────────────────────────
TEAM_ID = "TEAM001"
SUBMISSION_FILENAME = f"{TEAM_ID}_MM26ML03.csv"

# ── Real Dataset Feature Groups ────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "bp_unobtainable",
    "pain_score",
    "pain_assessable",
    "gcs_eye",
    "gcs_verbal",
    "gcs_motor",
    "gcs_total",
    "avpu_ordinal",
    "follows_commands",
    "vs_heartrate_min",
    "vs_heartrate_max",
    "vs_heartrate_mean",
    "vs_resprate_min",
    "vs_resprate_max",
    "vs_resprate_mean",
    "vs_sbp_min",
    "vs_sbp_max",
    "vs_sbp_mean",
    "vs_o2sat_min",
    "vs_o2sat_max",
    "vs_o2sat_mean",
    "vs_temperature_max",
    "n_vitalsign_readings",
    "n_diagnoses",
    "n_home_meds",
]

CATEGORICAL_FEATURES = [
    "gender",
    "race",
    "arrival_transport",
    "avpu",
    "consciousness_source",
    "chiefcomplaint",
]

# Clinical text narrative feature
TEXT_FEATURE = "description"

# All features (excluding identifier and target)
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]
