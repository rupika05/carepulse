"""
generate_evaluation_notebook.py — Programmatically builds and writes the complete
evaluation pipeline Jupyter Notebook for MM26ML03 Healthcare Triage Classification.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# MM26ML03 — Emergency Healthcare Triage Classification
### Complete Machine Learning Evaluation Pipeline & Cost-Sensitive Decision Optimization

**Competition Track:** `MM26ML03`  
**Target:** Predict emergency triage category (`RED`, `YELLOW`, `GREEN`, `BLACK`)  
**Triage System:** Simple Triage and Rapid Treatment (START) Protocol  
**Objective:** Minimize total clinical misclassification cost based on the competition cost matrix.

> **Clinical Disclaimer:** This system is a machine learning decision-support prototype. It is not an autonomous diagnostic tool and does not replace evaluation by licensed emergency medical professionals."""))

# Section 1
cells.append(nbf.v4.new_markdown_cell("""## 1. Problem Formulation & Exact Competition Cost Matrix

In disaster response and emergency triage, classification errors carry asymmetric clinical penalties:
- Misclassifying a critical patient (**RED**) as walking wounded (**GREEN**) results in severe, preventable morbidity or death (penalty = **10**).
- Over-triaging a walking wounded patient (**GREEN**) as immediate (**RED**) consumes resuscitation resources but does not directly harm the patient (penalty = **1**).

### Official Cost Matrix:
$$\\text{Cost Matrix} = \\begin{pmatrix}
0 & 5 & 10 & 5 \\\\
2 & 0 & 3 & 2 \\\\
1 & 1 & 0 & 1 \\\\
2 & 2 & 2 & 0
\\end{pmatrix}$$

Rows = Actual Class: `[RED, YELLOW, GREEN, BLACK]`  
Columns = Predicted Class: `[RED, YELLOW, GREEN, BLACK]`"""))

# Section 2: Imports & Environment
cells.append(nbf.v4.new_markdown_cell("""## 2. Environment Setup & Library Imports"""))

cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn & Gradient Boosting Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

print("Environment setup successful!")"""))

# Section 3: Data Loading & Schema Inspection
cells.append(nbf.v4.new_markdown_cell("""## 3. Dataset Loading & Schema Inspection"""))

cells.append(nbf.v4.new_code_cell("""# Define paths
data_dir = Path('../data')
train_path = data_dir / 'train.csv' if (data_dir / 'train.csv').exists() else data_dir / 'train_ml03.csv'
test_path = data_dir / 'test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Training dataset: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Test dataset:     {test_df.shape[0]} rows, {test_df.shape[1]} columns")

# Inspect schema
print("\\n--- Column Data Types ---")
print(train_df.dtypes.value_counts())

# Identify target and identifier
target_col = 'start_category'
id_col = 'subject_id'
print(f"\\nTarget Column:      {target_col}")
print(f"Identifier Column:  {id_col}")"""))

# Section 4: Missing Values & Class Distribution
cells.append(nbf.v4.new_markdown_cell("""## 4. Class Distribution & Missing Value Analysis"""))

cells.append(nbf.v4.new_code_cell("""CLASSES = ['RED', 'YELLOW', 'GREEN', 'BLACK']

# Class distribution
dist = train_df[target_col].value_counts()
dist_pct = train_df[target_col].value_counts(normalize=True) * 100

print("=== Class Distribution ===")
for cls in CLASSES:
    cnt = dist.get(cls, 0)
    pct = dist_pct.get(cls, 0)
    print(f"  {cls:<8}: {cnt:>4} ({pct:>5.2f}%)")

# Missing values
missing = train_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\\nTotal features with missing values: {len(missing)}")
print(missing.head(10))"""))

# Section 5: Exploratory Data Analysis
cells.append(nbf.v4.new_markdown_cell("""## 5. Exploratory Data Analysis & Visualizations"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Class Distribution Bar Chart
palette = {'RED': '#dc2626', 'YELLOW': '#ca8a04', 'GREEN': '#16a34a', 'BLACK': '#1e293b'}
sns.barplot(x=dist.index, y=dist.values, order=CLASSES, palette=palette, ax=axes[0])
axes[0].set_title('Triage Category Distribution in Cohort', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Patients')
for i, p in enumerate(axes[0].patches):
    axes[0].annotate(f"{int(p.get_height())}\\n({dist_pct[CLASSES[i]]:.1f}%)",
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold')

# 2. GCS Total vs Triage Class
sns.boxplot(x=target_col, y='gcs_total', data=train_df, order=CLASSES, palette=palette, ax=axes[1])
axes[1].set_title('Glasgow Coma Scale (GCS) Across Triage Categories', fontsize=12, fontweight='bold')
axes[1].set_ylabel('GCS Total Score (3 - 15)')

plt.tight_layout()
plt.show()"""))

# Section 6: Feature Preprocessing Pipeline (TF-IDF + Imputation + Scaling)
cells.append(nbf.v4.new_markdown_cell("""## 6. Multimodal Feature Preprocessing Pipeline

We construct a leak-free `ColumnTransformer`:
- **Numerical Features (31):** Median imputation $\\rightarrow$ StandardScaler
- **Categorical Features (6):** Most-frequent imputation $\\rightarrow$ OneHotEncoder
- **Clinical Narrative (1):** Missing fill $\\rightarrow$ TF-IDF Vectorizer (unigrams & bigrams)"""))

cells.append(nbf.v4.new_code_cell("""# Column specifications
NUMERICAL_FEATURES = [
    'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'bp_unobtainable',
    'pain_score', 'pain_assessable', 'gcs_eye', 'gcs_verbal', 'gcs_motor', 'gcs_total',
    'avpu_ordinal', 'follows_commands', 'vs_heartrate_min', 'vs_heartrate_max', 'vs_heartrate_mean',
    'vs_resprate_min', 'vs_resprate_max', 'vs_resprate_mean', 'vs_sbp_min', 'vs_sbp_max',
    'vs_sbp_mean', 'vs_o2sat_min', 'vs_o2sat_max', 'vs_o2sat_mean', 'vs_temperature_max',
    'n_vitalsign_readings', 'n_diagnoses', 'n_home_meds'
]

CATEGORICAL_FEATURES = [
    'gender', 'race', 'arrival_transport', 'avpu', 'consciousness_source', 'chiefcomplaint'
]

TEXT_FEATURE = 'description'

def _clean_text_series(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.Series(x).fillna('').astype(str)

def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    text_pipe = Pipeline([
        ('cleaner', FunctionTransformer(_clean_text_series, feature_names_out='one-to-one')),
        ('tfidf', TfidfVectorizer(max_features=60, ngram_range=(1, 2), stop_words='english', token_pattern=r'(?u)\\b[a-zA-Z]{3,}\\b'))
    ])
    
    return ColumnTransformer([
        ('num', num_pipe, NUMERICAL_FEATURES),
        ('cat', cat_pipe, CATEGORICAL_FEATURES),
        ('text', text_pipe, TEXT_FEATURE)
    ], remainder='drop')

print("Preprocessor configured with numerical, categorical, and TF-IDF text branches.")"""))

# Section 7: Cost-Sensitive Decision Layer Formulation
cells.append(nbf.v4.new_markdown_cell("""## 7. Cost-Sensitive Decision Layer Formulation

Given the predicted class probability distribution for patient $k$:
$$\\mathbf{p}_k = [P(\\text{RED}), P(\\text{YELLOW}), P(\\text{GREEN}), P(\\text{BLACK})]$$

The expected clinical cost for assigning patient $k$ to candidate triage category $j \\in \\{0, 1, 2, 3\\}$ is:
$$\\mathbb{E}[\\text{Cost}(j)] = \\sum_{i=0}^{3} p_{k, i} \\times C_{i, j}$$

The cost-sensitive decision selects:
$$\\hat{y}_k^* = \\arg\\min_{j} \\mathbb{E}[\\text{Cost}(j)]$$"""))

cells.append(nbf.v4.new_code_cell("""# Official Cost Matrix
COST_MATRIX = np.array([
    [0, 5, 10, 5],   # Actual RED
    [2, 0,  3, 2],   # Actual YELLOW
    [1, 1,  0, 1],   # Actual GREEN
    [2, 2,  2, 0]    # Actual BLACK
], dtype=float)

def compute_expected_costs(proba: np.ndarray) -> np.ndarray:
    \"\"\"Compute expected costs for candidate predictions: shape (n_samples, 4).\"\"\"
    # proba is (N, 4), COST_MATRIX is (4, 4) -> Expected cost for action j is proba @ COST_MATRIX[:, j]
    return np.dot(proba, COST_MATRIX)

def predict_cost_sensitive(proba: np.ndarray) -> np.ndarray:
    \"\"\"Select class minimizing expected cost for each sample.\"\"\"
    expected_costs = compute_expected_costs(proba)
    best_idx = np.argmin(expected_costs, axis=1)
    return np.array([CLASSES[idx] for idx in best_idx])

def calculate_total_cost(y_true: list, y_pred: list) -> float:
    \"\"\"Calculate total misclassification cost on a cohort.\"\"\"
    class_map = {c: i for i, c in enumerate(CLASSES)}
    total = 0.0
    for true, pred in zip(y_true, y_pred):
        total += COST_MATRIX[class_map[true], class_map[pred]]
    return total

print("Cost-sensitive decision framework initialized!")"""))

# Section 8: Model Training with 5-Fold Stratified Cross-Validation
cells.append(nbf.v4.new_markdown_cell("""## 8. Candidate Model Training & Stratified 5-Fold Cross-Validation

We train and evaluate 5 candidate architectures:
1. **Logistic Regression** (L2 regularized multiclass baseline)
2. **Random Forest** (Ensemble of decorrelated trees)
3. **Gradient Boosting** (Sequential gradient boosting on tabular + text)
4. **LightGBM** (Histogram-based fast gradient boosting)
5. **XGBoost** (Extreme gradient boosting with exact splitting)

Cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` preserving the class balance across every fold."""))

cells.append(nbf.v4.new_code_cell("""# Prepare X and y
X = train_df.drop(columns=[id_col, target_col])
y = train_df[target_col].values

# Candidate models
candidate_models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42, eval_metric='mlogloss', verbosity=0)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, clf in candidate_models.items():
    print(f">> Evaluating: {name} (5-Fold Stratified CV)...")
    pipe = Pipeline([
        ('prep', build_preprocessor()),
        ('clf', clf)
    ])
    
    # Predict Out-of-Fold probabilities
    oof_proba = cross_val_predict(pipe, X, y, cv=skf, method='predict_proba')
    
    # Ensure column order matches canonical CLASSES: [RED, YELLOW, GREEN, BLACK]
    unique_classes = sorted(list(np.unique(y)))
    reorder_idx = [unique_classes.index(c) for c in CLASSES]
    oof_proba = oof_proba[:, reorder_idx]
    
    # Apply Cost-Sensitive Decision Layer
    oof_preds = predict_cost_sensitive(oof_proba)
    
    # Calculate Metrics
    acc = accuracy_score(y, oof_preds)
    p, r, f1, _ = precision_recall_fscore_support(y, oof_preds, labels=CLASSES, average=None)
    macro_f1 = np.mean(f1)
    weighted_f1 = precision_recall_fscore_support(y, oof_preds, average='weighted')[2]
    total_cost = calculate_total_cost(y, oof_preds)
    
    cv_results[name] = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'total_cost': total_cost,
        'per_class_precision': dict(zip(CLASSES, p)),
        'per_class_recall': dict(zip(CLASSES, r)),
        'per_class_f1': dict(zip(CLASSES, f1)),
        'oof_proba': oof_proba,
        'oof_pred': oof_preds
    }

print("\\n[OK] Cross-validation completed across all 5 models!")"""))

# Section 9: Comprehensive Model Comparison
cells.append(nbf.v4.new_markdown_cell("""## 9. Model Comparison & Performance Analysis"""))

cells.append(nbf.v4.new_code_cell("""# 1. Overall Comparison Table
summary_rows = []
for name, res in cv_results.items():
    summary_rows.append({
        'Model': name,
        'Accuracy': f"{res['accuracy']:.4f}",
        'Macro-F1': f"{res['macro_f1']:.4f}",
        'Weighted-F1': f"{res['weighted_f1']:.4f}",
        'RED Recall': f"{res['per_class_recall']['RED']:.4f}",
        'Total Misclassification Cost': f"{res['total_cost']:.1f}"
    })

summary_df = pd.DataFrame(summary_rows)
print("=== Summary Performance Table ===")
display(summary_df)

# 2. Per-Class Metrics Table
print("\\n=== Per-Class Precision, Recall, and F1 Breakdown ===")
per_class_rows = []
for name, res in cv_results.items():
    for cls in CLASSES:
        per_class_rows.append({
            'Model': name,
            'Class': cls,
            'Precision': f"{res['per_class_precision'][cls]:.4f}",
            'Recall': f"{res['per_class_recall'][cls]:.4f}",
            'F1-Score': f"{res['per_class_f1'][cls]:.4f}"
        })
per_class_df = pd.DataFrame(per_class_rows)
display(per_class_df.head(10))"""))

# Section 10: Visual Comparison & Confusion Matrices
cells.append(nbf.v4.new_markdown_cell("""## 10. Visualizations: Model Comparison & Confusion Matrices"""))

cells.append(nbf.v4.new_code_cell("""# 1. Total Cost Comparison Chart
fig, ax = plt.subplots(figsize=(10, 4))
model_names = list(cv_results.keys())
costs = [cv_results[m]['total_cost'] for m in model_names]
colors = ['#3b82f6' if c > min(costs) else '#10b981' for c in costs]

bars = ax.bar(model_names, costs, color=colors, width=0.55)
ax.set_title('Total Misclassification Cost Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Cost')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.1f}", (bar.get_x() + bar.get_width() / 2, height + 3),
                ha='center', va='bottom', fontweight='bold')
plt.show()

# 2. Confusion Matrices for Top Models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
top_models = ['Logistic Regression', 'Gradient Boosting', 'LightGBM']

for i, m_name in enumerate(top_models):
    cm = confusion_matrix(y, cv_results[m_name]['oof_pred'], labels=CLASSES)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[i], cbar=False)
    axes[i].set_title(f"Confusion Matrix: {m_name}\\nCost: {cv_results[m_name]['total_cost']:.1f}", fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Predicted Triage')
    axes[i].set_ylabel('Actual Triage')

plt.tight_layout()
plt.show()"""))

# Section 11: Feature & Word Importances
cells.append(nbf.v4.new_markdown_cell("""## 11. Feature & Keyword Importance Analysis (Random Forest & Tree Ensembles)"""))

cells.append(nbf.v4.new_code_cell("""# Fit Random Forest on full training data to inspect feature importances
rf_pipe = Pipeline([
    ('prep', build_preprocessor()),
    ('clf', candidate_models['Random Forest'])
])
rf_pipe.fit(X, y)

feature_names = rf_pipe.named_steps['prep'].get_feature_names_out()
rf_importances = rf_pipe.named_steps['clf'].feature_importances_

# Separate physiological and text keyword importances
physio_imp = []
text_imp = []

for name, imp in zip(feature_names, rf_importances):
    if name.startswith('text__'):
        text_imp.append((name.replace('text__', ''), imp))
    else:
        clean_name = name.split('__')[-1]
        physio_imp.append((clean_name, imp))

physio_imp.sort(key=lambda x: x[1], reverse=True)
text_imp.sort(key=lambda x: x[1], reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Top Physiological Features
p_words, p_vals = zip(*physio_imp[:10])
sns.barplot(x=list(p_vals), y=list(p_words), palette='Blues_r', ax=axes[0])
axes[0].set_title('Top 10 Physiological Predictors (Random Forest)', fontweight='bold')
axes[0].set_xlabel('Feature Importance')

# Top Clinical Keywords Picked by Random Forest
t_words, t_vals = zip(*text_imp[:10])
sns.barplot(x=list(t_vals), y=list(t_words), palette='YlOrBr_r', ax=axes[1])
axes[1].set_title('Top 10 Clinical Keywords / Phrases Picked by Random Forest', fontweight='bold')
axes[1].set_xlabel('TF-IDF Feature Importance')

plt.tight_layout()
plt.show()"""))

# Section 12: Final Model Training & Verification
cells.append(nbf.v4.new_markdown_cell("""## 12. Final Clinical Model Selection & Full Training

We select the best-performing model based on:
1. **Lowest Total Misclassification Cost** (guaranteeing minimum expected clinical loss)
2. **High RED-class Recall** (critical to avoid missing immediate life threats)
3. **Macro-F1 Balance** across all 4 categories"""))

cells.append(nbf.v4.new_code_cell("""# Select model with lowest total cost
best_model_name = min(cv_results.keys(), key=lambda m: cv_results[m]['total_cost'])
best_metrics = cv_results[best_model_name]

print(f"=== Selected Champion Model: {best_model_name} ===")
print(f"  Total Cost:     {best_metrics['total_cost']:.1f}")
print(f"  Accuracy:       {best_metrics['accuracy']:.4f}")
print(f"  Macro-F1:       {best_metrics['macro_f1']:.4f}")
print(f"  Weighted-F1:    {best_metrics['weighted_f1']:.4f}")
print(f"  RED-Class Recall: {best_metrics['per_class_recall']['RED']:.4f}")

# Train champion model on 100% of training cohort
final_pipeline = Pipeline([
    ('prep', build_preprocessor()),
    ('clf', candidate_models[best_model_name])
])
final_pipeline.fit(X, y)
print(f"\\n[OK] Champion model ({best_model_name}) trained on full cohort ({len(X)} patients).")"""))

# Section 13: Submission Generation
cells.append(nbf.v4.new_markdown_cell("""## 13. Test Set Inference & Submission Generation

We generate the official competition submission:
- Preserves the exact patient ordering (`112` rows).
- Applies cost-sensitive decision minimization.
- Outputs class probabilities for `RED`, `YELLOW`, `GREEN`, `BLACK` summing to 1.0."""))

cells.append(nbf.v4.new_code_cell("""# Prepare test features
patient_ids = test_df[id_col].reset_index(drop=True)
X_test = test_df.drop(columns=[id_col], errors='ignore').copy()

# Ensure description exists
if 'description' not in X_test.columns:
    X_test['description'] = X_test['chiefcomplaint'].fillna('').astype(str) if 'chiefcomplaint' in X_test.columns else ''

# Predict probabilities
raw_proba = final_pipeline.predict_proba(X_test)
unique_classes = sorted(list(np.unique(y)))
reorder_idx = [unique_classes.index(c) for c in CLASSES]
raw_proba = raw_proba[:, reorder_idx]

# Cost-sensitive prediction
test_preds = predict_cost_sensitive(raw_proba)

# Create submission DataFrame
submission_df = pd.DataFrame({
    'Patient ID': patient_ids,
    'Predicted Triage': test_preds,
    'RED Probability': raw_proba[:, 0].round(6),
    'YELLOW Probability': raw_proba[:, 1].round(6),
    'GREEN Probability': raw_proba[:, 2].round(6),
    'BLACK Probability': raw_proba[:, 3].round(6),
})

# Validation Checks
assert len(submission_df) == len(test_df), "Row count mismatch!"
assert set(submission_df['Predicted Triage']).issubset(set(CLASSES)), "Invalid labels!"
prob_sums = submission_df[['RED Probability', 'YELLOW Probability', 'GREEN Probability', 'BLACK Probability']].sum(axis=1)
assert np.allclose(prob_sums, 1.0, atol=1e-3), "Probabilities do not sum to 1.0!"

# Save submission
output_dir = Path('../outputs')
output_dir.mkdir(parents=True, exist_ok=True)
sub_path = output_dir / 'TEAM001_MM26ML03.csv'
submission_df.to_csv(sub_path, index=False)

print(f"[OK] Submission validated and saved to: {sub_path}")
print(f"Distribution of test predictions:")
print(submission_df['Predicted Triage'].value_counts())
print("\\nFirst 5 Rows of Submission:")
display(submission_df.head(5))"""))

# Assign cells to notebook
nb.cells = cells

# Save notebook
notebook_path = Path("notebooks/triage_model_development.ipynb")
notebook_path.parent.mkdir(parents=True, exist_ok=True)
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated complete evaluation notebook at {notebook_path} with {len(cells)} cells!")
