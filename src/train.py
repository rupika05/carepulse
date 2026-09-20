"""
train.py — Model training script for the Healthcare Triage Classification project.

Trains multiple classifiers (including XGBoost & LightGBM), evaluates each
using stratified cross-validation, selects the best model based on total
misclassification cost, and saves the final pipeline (preprocessor + model).

Key improvements:
  - class_weight='balanced' for models that support it
  - CalibratedClassifierCV for well-calibrated probabilities
  - XGBoost and LightGBM included alongside sklearn models
  - Detailed prediction distribution diagnostics

Usage:
    python -m src.train
"""
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

from src.config import (
    RANDOM_SEED, CLASSES, CV_FOLDS, MODEL_PATH, MODELS_DIR, OUTPUTS_DIR
)
from src.data_processing import load_train, inspect_dataset, validate_column_match, load_test, split_features_target
from src.preprocessing import build_preprocessor, encode_labels, decode_labels
from src.evaluate import compute_all_metrics, print_metrics_table, plot_confusion_matrix, save_metrics
from src.cost_sensitive import predict_cost_sensitive_labels, compute_expected_costs_batch


# ── Candidate models ──────────────────────────────────────────────────────────
def get_candidate_models() -> dict:
    """
    Return all candidate models. Uses class_weight='balanced' where supported
    to counteract the class imbalance (YELLOW >> BLACK).
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            random_state=RANDOM_SEED,
        ),
    }

    # XGBoost — with sample_weight balancing via scale_pos_weight
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    except ImportError:
        print("[WARN] XGBoost not installed, skipping.")

    # LightGBM — with class_weight='balanced'
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        print("[WARN] LightGBM not installed, skipping.")

    return models


def _compute_sample_weights(y: pd.Series) -> np.ndarray:
    """Compute balanced sample weights for models that don't support class_weight."""
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def train_and_evaluate_all(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train all candidate models using stratified cross-validation.
    Returns dict mapping model_name -> {metrics, oof_proba, oof_pred}.
    """
    preprocessor = build_preprocessor()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    for model_name, clf in get_candidate_models().items():
        print(f"\n>> Evaluating: {model_name}")

        # Wrap with CalibratedClassifierCV for well-calibrated probabilities
        calibrated_clf = CalibratedClassifierCV(clf, cv=3, method="sigmoid")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", calibrated_clf),
        ])

        # Get out-of-fold probability predictions
        oof_proba = cross_val_predict(
            pipeline, X, y,
            cv=skf, method="predict_proba"
        )

        # Reorder columns from sklearn alphabetical order to canonical CLASSES order
        unique_classes = sorted(np.unique(y))  # sklearn uses alphabetical
        reorder_idx = [unique_classes.index(cls) for cls in CLASSES]
        oof_proba = oof_proba[:, reorder_idx]

        # Apply cost-sensitive decision layer
        y_pred_cost = predict_cost_sensitive_labels(oof_proba)

        metrics = compute_all_metrics(list(y), y_pred_cost, model_name=model_name)
        results[model_name] = {
            "metrics": metrics,
            "oof_proba": oof_proba,
            "oof_pred": y_pred_cost,
        }

        # Print prediction distribution for diagnostics
        pred_dist = pd.Series(y_pred_cost).value_counts()
        actual_dist = y.value_counts()

        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Macro-F1:     {metrics['macro_f1']:.4f}")
        print(f"  Weighted-F1:  {metrics['weighted_f1']:.4f}")
        print(f"  Total Cost:   {metrics['total_cost']:.1f}")
        print(f"  Pred. dist:   {dict(pred_dist)}")
        print(f"  Mean proba:   " + ", ".join(
            f"{cls}={oof_proba[:, i].mean():.3f}" for i, cls in enumerate(CLASSES)
        ))

    return results


def select_best_model(results: dict) -> str:
    """Select model with lowest total misclassification cost."""
    best = min(results.keys(), key=lambda k: results[k]["metrics"]["total_cost"])
    print(f"\n[OK] Best model (lowest cost): {best}")
    return best


def train_final_pipeline(X: pd.DataFrame, y: pd.Series, model_name: str):
    """Train the selected model on the full training set with calibration."""
    models = get_candidate_models()
    clf = models[model_name]

    # Wrap with calibration for well-calibrated probabilities
    calibrated_clf = CalibratedClassifierCV(clf, cv=3, method="sigmoid")

    preprocessor = build_preprocessor()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated_clf),
    ])

    print(f"\n>> Training final pipeline ({model_name}) on full training set...")
    pipeline.fit(X, y)

    # Diagnostic: check train-set probability calibration
    train_proba = pipeline.predict_proba(X)
    pipeline_classes = list(pipeline.classes_)
    if pipeline_classes != CLASSES:
        reorder_idx = [pipeline_classes.index(cls) for cls in CLASSES]
        train_proba = train_proba[:, reorder_idx]
    train_preds = predict_cost_sensitive_labels(train_proba)
    train_dist = pd.Series(train_preds).value_counts()
    print(f"  Train prediction distribution: {dict(train_dist)}")
    print(f"  Train mean proba: " + ", ".join(
        f"{cls}={train_proba[:, i].mean():.3f}" for i, cls in enumerate(CLASSES)
    ))

    print("[OK] Training complete.")
    return pipeline


def extract_and_display_rf_words(pipeline, top_n: int = 20) -> list:
    """Extract and display the top predictive words/keywords learned by Random Forest."""
    import json
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Handle CalibratedClassifierCV wrapper
    actual_clf = classifier
    if hasattr(classifier, "estimator"):
        actual_clf = classifier.estimator
    # For calibrated classifiers, get the base estimator's feature importances
    if hasattr(actual_clf, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        importances = actual_clf.feature_importances_
    elif hasattr(classifier, "calibrated_classifiers_"):
        # Average importances across calibrated sub-estimators
        feature_names = preprocessor.get_feature_names_out()
        all_importances = []
        for cc in classifier.calibrated_classifiers_:
            if hasattr(cc.estimator, "feature_importances_"):
                all_importances.append(cc.estimator.feature_importances_)
        if all_importances:
            importances = np.mean(all_importances, axis=0)
        else:
            return []
    else:
        return []

    text_importances = []
    for name, imp in zip(feature_names, importances):
        if name.startswith("text__"):
            word = name.replace("text__", "")
            text_importances.append({"word": word, "importance": round(float(imp), 6)})

    text_importances.sort(key=lambda x: x["importance"], reverse=True)

    print(f"\n=== Top {min(top_n, len(text_importances))} Words / Phrases Picked by Random Forest ===")
    print(f"{'Keyword / Clinical Phrase':<30} {'RF Importance':>15}")
    print("-" * 47)
    for item in text_importances[:top_n]:
        print(f"{item['word']:<30} {item['importance']:>15.5f}")

    # Save to outputs directory
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    word_file = OUTPUTS_DIR / "rf_word_importances.json"
    with open(word_file, "w") as f:
        json.dump(text_importances, f, indent=2)
    print(f"[OK] Word importances saved to {word_file}")

    return text_importances


def save_pipeline(pipeline, model_name: str) -> None:
    """Save the fitted pipeline to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "model_name": model_name,
        "classes": CLASSES,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"[OK] Model saved to {MODEL_PATH}")


def main():
    print("=" * 60)
    print("  Healthcare Triage Classification -- Model Training")
    print("=" * 60)

    # Load & inspect data
    train_df = load_train()
    test_df = load_test()
    inspect_dataset(train_df, name="Training Data")
    validate_column_match(train_df, test_df)

    # Split features / target
    X, y = split_features_target(train_df)

    # Cross-validate all models
    print("\n=== Cross-Validation Evaluation ===")
    results = train_and_evaluate_all(X, y)

    # Print comparison table
    metrics_list = [v["metrics"] for v in results.values()]
    print_metrics_table(metrics_list)

    # Select best
    best_name = select_best_model(results)
    best_results = results[best_name]

    # Plot confusion matrix for best model (OOF predictions)
    plot_confusion_matrix(
        list(y),
        best_results["oof_pred"],
        model_name=f"{best_name} (OOF)"
    )

    # Save metrics
    save_metrics(best_results["metrics"])

    # Train final pipeline on full training data
    final_pipeline = train_final_pipeline(X, y, best_name)
    save_pipeline(final_pipeline, best_name)

    # Extract word importances from Random Forest
    if best_name == "Random Forest":
        extract_and_display_rf_words(final_pipeline)
    else:
        print("\n>> Extracting word importances from Random Forest...")
        rf_pipeline = train_final_pipeline(X, y, "Random Forest")
        extract_and_display_rf_words(rf_pipeline)

    print("\n=== Training Complete ===")
    print(f"Best model:   {best_name}")
    print(f"Accuracy:     {best_results['metrics']['accuracy']:.4f}")
    print(f"Macro-F1:     {best_results['metrics']['macro_f1']:.4f}")
    print(f"Weighted-F1:  {best_results['metrics']['weighted_f1']:.4f}")
    print(f"Total Cost:   {best_results['metrics']['total_cost']:.1f}")


if __name__ == "__main__":
    main()
