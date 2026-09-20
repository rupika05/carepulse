"""
run_full_evaluation.py — Complete execution script to evaluate all 5 models,
calculate metrics, compute cost-sensitive predictions, and output results.
"""
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings('ignore')
np.random.seed(42)

sys.path.append('.')
from src.config import CLASSES, COST_MATRIX, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURE
from src.preprocessing import build_preprocessor

# 1. Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

X = train_df.drop(columns=['subject_id', 'start_category'])
y = train_df['start_category'].values

print(f"Loaded train: {X.shape}, test: {test_df.shape}")
print(f"Target distribution:\n{pd.Series(y).value_counts()}\n")

# 2. Cost-sensitive helper functions
cost_mat = np.array(COST_MATRIX, dtype=float)

def predict_cost_sensitive(proba):
    expected_costs = np.dot(proba, cost_mat)
    best_idx = np.argmin(expected_costs, axis=1)
    return np.array([CLASSES[idx] for idx in best_idx])

def calculate_total_cost(y_true, y_pred):
    class_map = {c: i for i, c in enumerate(CLASSES)}
    return sum(cost_mat[class_map[t], class_map[p]] for t, p in zip(y_true, y_pred))

# 3. Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42, eval_metric='mlogloss', verbosity=0)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("=== Starting 5-Fold Stratified Cross-Validation ===")
for name, clf in models.items():
    pipe = Pipeline([
        ('prep', build_preprocessor()),
        ('clf', clf)
    ])
    
    oof_proba = cross_val_predict(pipe, X, y, cv=skf, method='predict_proba')
    
    unique_classes = sorted(list(np.unique(y)))
    reorder_idx = [unique_classes.index(c) for c in CLASSES]
    oof_proba = oof_proba[:, reorder_idx]
    
    oof_preds = predict_cost_sensitive(oof_proba)
    
    acc = accuracy_score(y, oof_preds)
    p, r, f1, _ = precision_recall_fscore_support(y, oof_preds, labels=CLASSES, average=None)
    macro_f1 = np.mean(f1)
    weighted_f1 = precision_recall_fscore_support(y, oof_preds, average='weighted')[2]
    total_cost = calculate_total_cost(y, oof_preds)
    
    results[name] = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'total_cost': total_cost,
        'precision': dict(zip(CLASSES, p)),
        'recall': dict(zip(CLASSES, r)),
        'f1': dict(zip(CLASSES, f1)),
        'oof_proba': oof_proba,
        'oof_preds': oof_preds
    }
    print(f"  {name:<22} | Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Cost: {total_cost:.1f} | RED Recall: {r[0]:.4f}")

# 4. Summary Table
print("\n" + "=" * 75)
print(f"{'Model':<22} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12} {'Total Cost':>12}")
print("-" * 75)
for name, res in results.items():
    print(f"{name:<22} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f} {res['weighted_f1']:>12.4f} {res['total_cost']:>12.1f}")
print("=" * 75)

# 5. Best Model Per-Class Breakdown
best_name = min(results.keys(), key=lambda m: results[m]['total_cost'])
best = results[best_name]

print(f"\n=== Champion Model: {best_name} (Lowest Misclassification Cost) ===")
print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 45)
for c in CLASSES:
    print(f"{c:<10} {best['precision'][c]:>10.4f} {best['recall'][c]:>10.4f} {best['f1'][c]:>10.4f}")
print("-" * 45)

# Confusion Matrix for best model
cm = confusion_matrix(y, best['oof_preds'], labels=CLASSES)
print("\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
print(f"{'Actual \\ Pred':<15} " + " ".join(f"{c:>8}" for c in CLASSES))
for idx, c in enumerate(CLASSES):
    print(f"{c:<15} " + " ".join(f"{cm[idx, j]:>8}" for j in range(4)))

# 6. Fit best model on full data & generate test submission
print(f"\n>> Training final {best_name} pipeline on 100% of data...")
final_pipe = Pipeline([
    ('prep', build_preprocessor()),
    ('clf', models[best_name])
])
final_pipe.fit(X, y)

X_test = test_df.drop(columns=['subject_id'], errors='ignore').copy()
if 'description' not in X_test.columns:
    X_test['description'] = X_test['chiefcomplaint'].fillna('').astype(str) if 'chiefcomplaint' in X_test.columns else ''

raw_test_proba = final_pipe.predict_proba(X_test)
unique_classes = sorted(list(np.unique(y)))
reorder_idx = [unique_classes.index(c) for c in CLASSES]
raw_test_proba = raw_test_proba[:, reorder_idx]

test_preds = predict_cost_sensitive(raw_test_proba)

sub_df = pd.DataFrame({
    'Patient ID': test_df['subject_id'],
    'Predicted Triage': test_preds,
    'RED Probability': raw_test_proba[:, 0].round(6),
    'YELLOW Probability': raw_test_proba[:, 1].round(6),
    'GREEN Probability': raw_test_proba[:, 2].round(6),
    'BLACK Probability': raw_test_proba[:, 3].round(6),
})

sub_path = Path('outputs/TEAM001_MM26ML03.csv')
sub_df.to_csv(sub_path, index=False)
print(f"\n[OK] Submission saved to {sub_path} ({len(sub_df)} patients)")
print("Test set triage distribution:")
for c in CLASSES:
    cnt = (sub_df['Predicted Triage'] == c).sum()
    print(f"  {c:<8}: {cnt:>3} ({cnt/len(sub_df)*100:.1f}%)")
