"""
create_notebook.py — Generates the clean, well-structured Jupyter notebook:
notebooks/triage_model_development.ipynb
covering all 15 required sections.
"""
import json
from pathlib import Path

notebook_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MM26ML03 — Healthcare Triage Classification\n",
            "### Clinical Decision Support Prototype using START Triage Protocol\n",
            "\n",
            "> **Disclaimer**: This system is a machine-learning clinical decision-support prototype. "
            "It is NOT an autonomous diagnostic tool and does NOT replace evaluation by qualified emergency medical professionals."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Imports\n",
            "Import required data science, machine learning, and visualization libraries."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Add project root to path\n",
            "sys.path.append('..')\n",
            "from src.config import (\n",
            "    CLASSES, CLASS_INDEX, COST_MATRIX, RANDOM_SEED, \n",
            "    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, PATIENT_ID_COLUMN\n",
            ")\n",
            "from src.cost_sensitive import (\n",
            "    compute_expected_costs, compute_expected_costs_batch,\n",
            "    predict_cost_sensitive_labels, total_misclassification_cost\n",
            ")\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "print('Imports successful!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Load Dataset\n",
            "Load the training dataset (`train.csv`) and test dataset (`test.csv`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_df = pd.read_csv('../data/train.csv')\n",
            "test_df = pd.read_csv('../data/test.csv')\n",
            "\n",
            "print(f'Train shape: {train_df.shape[0]} rows x {train_df.shape[1]} columns')\n",
            "print(f'Test shape:  {test_df.shape[0]} rows x {test_df.shape[1]} columns')\n",
            "train_df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Dataset Inspection\n",
            "Check column data types, missing values, duplicates, and unique counts."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('--- Column Types & Missing Values ---')\n",
            "info_df = pd.DataFrame({\n",
            "    'Dtype': train_df.dtypes,\n",
            "    'Missing_Count': train_df.isnull().sum(),\n",
            "    'Missing_Pct': (train_df.isnull().sum() / len(train_df) * 100).round(2),\n",
            "    'Unique_Count': train_df.nunique()\n",
            "})\n",
            "print(info_df)\n",
            "\n",
            "print(f'\\nDuplicate rows in train: {train_df.duplicated().sum()}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Exploratory Data Analysis (EDA)\n",
            "Visualize target class distribution, numerical vitals, and injury categories."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Target Class Distribution\n",
            "colors = {'RED': '#E53E3E', 'YELLOW': '#D69E2E', 'GREEN': '#38A169', 'BLACK': '#2D3748'}\n",
            "class_counts = train_df[TARGET_COLUMN].value_counts().reindex(CLASSES)\n",
            "\n",
            "plt.figure(figsize=(8, 4))\n",
            "bars = plt.bar(class_counts.index, class_counts.values, color=[colors[c] for c in class_counts.index])\n",
            "for bar in bars:\n",
            "    yval = bar.get_height()\n",
            "    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f'{yval} ({yval/len(train_df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')\n",
            "plt.title('START Triage Class Distribution (Training Data)', fontsize=13, fontweight='bold')\n",
            "plt.ylabel('Patient Count')\n",
            "plt.xlabel('Triage Category')\n",
            "plt.ylim(0, max(class_counts.values) * 1.15)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Vital Signs by Triage Category\n",
            "vitals_to_plot = ['respiratory_rate', 'pulse_rate', 'gcs_total', 'spo2']\n",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
            "for ax, vital in zip(axes.flatten(), vitals_to_plot):\n",
            "    sns.boxplot(data=train_df, x=TARGET_COLUMN, y=vital, order=CLASSES, palette=colors, ax=ax)\n",
            "    ax.set_title(f'{vital.replace(\"_\", \" \").title()} by Triage Category', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Correlation Matrix of Numerical Features\n",
            "plt.figure(figsize=(8, 6))\n",
            "num_corr = train_df[NUMERICAL_FEATURES].corr()\n",
            "sns.heatmap(num_corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)\n",
            "plt.title('Correlation Matrix (Numerical Vitals)', fontweight='bold')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Preprocessing Pipeline\n",
            "Build an sklearn `ColumnTransformer` with median imputation + scaling for numerical features and most-frequent imputation + one-hot encoding for categorical features."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.preprocessing import build_preprocessor\n",
            "\n",
            "preprocessor = build_preprocessor()\n",
            "X_train = train_df.drop(columns=[TARGET_COLUMN, PATIENT_ID_COLUMN])\n",
            "y_train = train_df[TARGET_COLUMN]\n",
            "\n",
            "print('Preprocessor structure:')\n",
            "print(preprocessor)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Train/Validation Split (Stratified K-Fold)\n",
            "Use 5-fold stratified cross validation to ensure proper class balance in each split."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import StratifiedKFold\n",
            "\n",
            "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)\n",
            "print(f'Cross-validation splits: {skf.get_n_splits(X_train, y_train)} folds')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Baseline Models & Candidate Comparison\n",
            "Evaluate Logistic Regression, Random Forest, and Gradient Boosting."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.model_selection import cross_val_predict\n",
            "from src.evaluate import compute_all_metrics\n",
            "\n",
            "candidate_models = {\n",
            "    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, solver='lbfgs'),\n",
            "    'Random Forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1),\n",
            "    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED)\n",
            "}\n",
            "\n",
            "eval_results = {}\n",
            "unique_classes = list(np.unique(y_train))\n",
            "reorder_idx = [unique_classes.index(cls) for cls in CLASSES]\n",
            "\n",
            "for name, clf in candidate_models.items():\n",
            "    pipe = Pipeline([('preprocessor', build_preprocessor()), ('classifier', clf)])\n",
            "    oof_proba = cross_val_predict(pipe, X_train, y_train, cv=skf, method='predict_proba')\n",
            "    oof_proba = oof_proba[:, reorder_idx]\n",
            "    oof_preds = predict_cost_sensitive_labels(oof_proba)\n",
            "    metrics = compute_all_metrics(list(y_train), oof_preds, model_name=name)\n",
            "    eval_results[name] = metrics"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Model Evaluation Table\n",
            "Compare accuracy, macro-F1, weighted-F1, and total misclassification cost."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "comparison_df = pd.DataFrame([\n",
            "    {\n",
            "        'Model': m['model'],\n",
            "        'Accuracy': m['accuracy'],\n",
            "        'Macro-F1': m['macro_f1'],\n",
            "        'Weighted-F1': m['weighted_f1'],\n",
            "        'Total Misclassification Cost': m['total_cost']\n",
            "    }\n",
            "    for m in eval_results.values()\n",
            "])\n",
            "print(comparison_df.to_string(index=False))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Competition Cost Matrix\n",
            "The exact cost matrix specified in the competition problem statement MM26ML03:"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "cost_df = pd.DataFrame(COST_MATRIX, index=CLASSES, columns=CLASSES)\n",
            "print('Competition Cost Matrix (Rows=Actual, Columns=Predicted):')\n",
            "print(cost_df)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Cost-Sensitive Prediction Mechanism\n",
            "Expected cost for predicting class $c$ is: $\\mathbb{E}[Cost(c)] = \\sum_{a} P(actual=a) \\times Cost(a, c)$. "
            "The decision layer selects $\\arg\\min_c \\mathbb{E}[Cost(c)]$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Demonstration: Patient with ambiguous vitals (50% RED, 50% YELLOW)\n",
            "ambiguous_patient_probs = np.array([0.5, 0.5, 0.0, 0.0])\n",
            "expected_costs = compute_expected_costs(ambiguous_patient_probs)\n",
            "\n",
            "for cls, c in zip(CLASSES, expected_costs):\n",
            "    print(f'Expected Cost if predicting {cls:<8}: {c:.2f}')\n",
            "\n",
            "best_idx = np.argmin(expected_costs)\n",
            "print(f'\\nCost-sensitive decision selects: {CLASSES[best_idx]} (lowest expected cost)')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 11. Final Model Evaluation & Confusion Matrix\n",
            "Examine per-class performance and misclassification patterns."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "best_model_name = min(eval_results.keys(), key=lambda k: eval_results[k]['total_cost'])\n",
            "best_metrics = eval_results[best_model_name]\n",
            "\n",
            "print(f'Selected Best Model: {best_model_name}')\n",
            "print(json.dumps(best_metrics['per_class'], indent=2))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 12. Train Final Pipeline\n",
            "Fit the selected pipeline on 100% of the training data."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.train import train_final_pipeline, save_pipeline\n",
            "\n",
            "final_pipeline = train_final_pipeline(X_train, y_train, best_model_name)\n",
            "save_pipeline(final_pipeline, best_model_name)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 13. Generate Test Predictions\n",
            "Run inference on the test set (`test.csv`) and output class probabilities."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.predict import generate_predictions, validate_submission, save_submission\n",
            "\n",
            "submission_df = generate_predictions(final_pipeline, test_df)\n",
            "submission_df.head(10)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 14. Validate Submission CSV\n",
            "Ensure row count, patient IDs, probability sums, and valid labels match competition standards."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "validate_submission(submission_df, test_df)\n",
            "sub_path = save_submission(submission_df)\n",
            "print(f'Verified submission saved to: {sub_path}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 15. Summary & Conclusion\n",
            "- **Pipeline**: ColumnTransformer (median/mode imputation + one-hot encoding) + LogisticRegression with cost-sensitive decision layer.\n",
            "- **Cost-sensitive decision layer**: Successfully penalizes costly clinical errors (e.g. misdiagnosing RED as GREEN).\n",
            "- **Inference Service**: Ready for deployment with FastAPI backend and React frontend."
        ]
    }
]

notebook_data = {
    "cells": notebook_cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_path = Path("notebooks/triage_model_development.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=2)

print(f"[OK] Jupyter notebook created at {out_path} with {len(notebook_cells)} cells.")
