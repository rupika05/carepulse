# Healthcare Triage Classification (MM26ML03)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5+-646CFF.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning clinical decision-support system for emergency triage classification under the **Simple Triage and Rapid Treatment (START)** framework, incorporating an **unequal-penalty cost matrix decision layer** and an emergency-response web dashboard.

---

## 1. Problem Statement

In mass casualty incidents (MCI) and emergency departments, high patient volumes arrive concurrently, demanding rapid, objective acuity assessment. Under-triaging a critical casualty can be fatal, while over-triaging misallocates scarce ICU beds and surgical suites. 

Standard machine learning classifiers treat all classification errors equally. However, in emergency care:
- **Misclassifying RED (Immediate) as GREEN (Minor)** is potentially catastrophic.
- **Misclassifying GREEN (Minor) as YELLOW (Delayed)** has minimal clinical risk.

This project implements an end-to-end cost-sensitive machine learning pipeline that minimizes expected clinical cost rather than pure classification error.

---

## 2. Objective

Classify emergency casualties into four standardized START triage categories:

| Triage Code | Category | Clinical Description | Priority |
|---|---|---|---|
| **RED** | Immediate | Life-threatening conditions (compromised airway, severe shock, altered mental status) requiring urgent intervention | 1 |
| **YELLOW** | Delayed | Serious injuries requiring medical care, but stable vitals permit delayed treatment | 2 |
| **GREEN** | Minor | &ldquo;Walking wounded&rdquo; — ambulatory patients with minor/superficial injuries | 3 |
| **BLACK** | Expectant / Deceased | Patients who are deceased or have catastrophic trauma incompatible with survival | 4 |

---

## 3. Dataset Overview

The dataset is clinically grounded in the official START triage protocol and features physiological, neurological, demographic, and injury parameters:

- **Training Records**: 3,000 patients
- **Test Records**: 750 patients
- **Class Distribution (Train)**:
  - `YELLOW`: 1,085 (36.2%)
  - `RED`: 859 (28.6%)
  - `GREEN`: 748 (24.9%)
  - `BLACK`: 308 (10.3%)
- **Target Column**: `triage_label` (`RED`, `YELLOW`, `GREEN`, `BLACK`)
- **Patient Identifier**: `patient_id` (excluded from training to prevent leakage)

### Feature Schema:
- **Numerical (7)**: `age`, `respiratory_rate`, `pulse_rate`, `systolic_bp`, `spo2`, `gcs_total`, `capillary_refill_sec`
- **Categorical (9)**: `gender`, `avpu`, `respiratory_effort`, `radial_pulse_present`, `major_hemorrhage`, `ambulatory`, `injury_type`, `injury_severity`, `mechanism_of_injury`

---

## 4. Approach & Pipeline

```
Raw Patient Features
       ↓
Median / Mode Imputation (fit on train only)
       ↓
StandardScaler (numerical) & OneHotEncoder (categorical)
       ↓
Classification Pipeline (Calibrated Probabilities)
       ↓
Cost-Sensitive Decision Layer (Expected Cost Minimization)
       ↓
Final Triage Recommendation + Decision Analytics
```

1. **Preprocessing**: Handled via scikit-learn `ColumnTransformer`. Fitted exclusively on training data to prevent information leakage. Numerical features use median imputation and standard scaling; categorical features use most-frequent imputation and one-hot encoding.
2. **Stratified Validation**: Evaluated using 5-Fold Stratified Cross-Validation (`StratifiedKFold`) preserving class balance.
3. **Candidate Models Evaluated**:
   - Logistic Regression (`lbfgs` multiclass)
   - Random Forest (200 estimators, min_samples_leaf=2)
   - Gradient Boosting (200 estimators, learning_rate=0.1, max_depth=5)
4. **Final Model**: Logistic Regression pipeline combined with the expected-cost decision layer, offering superior probability calibration and zero misclassification cost on validation.

---

## 5. Cost-Sensitive Decision Mechanism

The competition specifies the following exact cost matrix $\mathbf{C} \in \mathbb{R}^{4 \times 4}$:

```
                   PREDICTED
ACTUAL        RED   YELLOW   GREEN   BLACK

RED            0      5       10       5
YELLOW         2      0        3       2
GREEN          1      1        0       1
BLACK          2      2        2       0
```

### Expected Cost Formula:
For an arriving patient with model posterior probabilities $P(\text{actual} = a \mid \mathbf{x})$, the expected cost of selecting predicted class $c$ is:

$$\mathbb{E}[\text{Cost}(c)] = \sum_{a \in \{\text{RED, YELLOW, GREEN, BLACK}\}} P(\text{actual} = a \mid \mathbf{x}) \times C(a, c)$$

In matrix form:
$$\mathbf{e} = \mathbf{p} \times \mathbf{C}$$

The decision layer selects the class with the minimum expected cost:
$$\hat{y} = \arg\min_{c} \mathbb{E}[\text{Cost}(c)]$$

This ensures the model aggressively protects high-risk casualties (e.g. patients with border-line vitals are steered to RED rather than GREEN).

---

## 6. Evaluation Metrics

### Cross-Validation Model Comparison:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Total Cost |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression (Selected)** | **1.0000** | **1.0000** | **1.0000** | **0.0** |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 0.0 |
| **Gradient Boosting** | 0.9997 | 0.9997 | 0.9997 | 5.0 |

### Per-Class Metrics (Final Model):
- **RED**: Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 859
- **YELLOW**: Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 1085
- **GREEN**: Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 748
- **BLACK**: Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | Support: 308

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                   │
│   - Triage Dashboard                                        │
│   - Patient Clinical Assessment Form (16 standard features) │
│   - 1-Click Clinical Demo Presets                           │
│   - Cost-Sensitive Decision Visualizer                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON (REST)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Python)                 │
│   - GET /health                                             │
│   - POST /predict (Pydantic schema validation)              │
│   - Model loaded once at startup (Lifespan pattern)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Inference Engine (joblib)                  │
│   - Preprocessing: ColumnTransformer                        │
│   - Classifier: LogisticRegression (predict_proba)          │
│   - Decision: Expected Cost Minimization Layer              │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Local Setup & Installation

### Prerequisites:
- Python 3.10+
- Node.js 18+ and npm

### 1. Clone the repository & install Python dependencies:
```bash
git clone <repo-url>
cd healthcare_traige

# Install Python requirements
python -m pip install -r requirements.txt
```

### 2. Run tests to verify setup:
```bash
python -m pytest tests/ -v
```

---

## 9. Running the Project

### Start the Backend (FastAPI):
```bash
# From project root:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at: `http://localhost:8000/docs`

### Start the Frontend (React + Vite):
```bash
# In a second terminal:
cd frontend
npm install
npm run dev
```
Open your browser at: `http://localhost:5173`

---

## 10. Reproducibility & Pipeline Scripts

- **Generate Dataset**: `python generate_dataset.py`
- **Train & Evaluate Models**: `python -u -m src.train`
- **Generate Competition Submission**: `python -u -m src.predict`
- **Run Full Test Suite**: `python -m pytest tests/ -v`

---

## 11. Competition Submission File

The final competition submission is generated at:
```
outputs/TEAM001_MM26ML03.csv
```

### Format Verification:
- **Total Rows**: Exactly 750 rows (matches test set).
- **Patient IDs**: Preserves exact test set IDs (`PAT_03001` to `PAT_03750`).
- **Ordering**: Unchanged from input `test.csv`.
- **Probabilities**: Includes `P_RED`, `P_YELLOW`, `P_GREEN`, `P_BLACK` (all $\in [0, 1]$ and summing to 1.0).
- **Class Labels**: Restricted to `RED`, `YELLOW`, `GREEN`, `BLACK`.

Sample rows:
```csv
patient_id,Predicted_Triage,P_RED,P_YELLOW,P_GREEN,P_BLACK
PAT_03001,RED,0.997965,0.000384,0.000001,0.001651
PAT_03002,RED,0.990372,0.005556,0.000002,0.004070
PAT_03003,YELLOW,0.010967,0.988905,0.000126,0.000002
PAT_03004,GREEN,0.000001,0.002876,0.997124,0.000000
PAT_03005,BLACK,0.011011,0.000000,0.000000,0.988989
```

---

## 12. Clinical Decision-Support Disclaimer

> **IMPORTANT CLINICAL NOTICE**  
> This system is a machine learning research and clinical decision-support prototype developed for competition MM26ML03.  
> It is **NOT** an autonomous diagnostic device, does **NOT** constitute medical advice, and is **NOT** a replacement for licensed emergency medical personnel or institutional triage protocols. All triage classifications must be verified by a qualified medical professional.
