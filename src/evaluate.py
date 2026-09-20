"""
evaluate.py — Evaluation metrics for the triage classification system.

Computes:
  - Accuracy
  - Per-class precision, recall, F1
  - Macro-F1, Weighted-F1
  - Confusion matrix
  - Total misclassification cost
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from pathlib import Path
from typing import Dict, Any, List

from src.config import CLASSES, OUTPUTS_DIR, CONFUSION_MATRIX_PATH, METRICS_PATH
from src.cost_sensitive import total_misclassification_cost


# Triage category colours (accessible palette)
CLASS_COLORS = {
    "RED": "#E53E3E",
    "YELLOW": "#D69E2E",
    "GREEN": "#38A169",
    "BLACK": "#2D3748",
}


def compute_all_metrics(
    y_true: List[str],
    y_pred: List[str],
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Compute the full evaluation metric suite required by the competition.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
    cost = total_misclassification_cost(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, labels=CLASSES,
        output_dict=True, zero_division=0
    )

    per_class = {}
    for cls in CLASSES:
        per_class[cls] = {
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1": round(report[cls]["f1-score"], 4),
            "support": int(report[cls]["support"]),
        }

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_cost": round(cost, 2),
        "per_class": per_class,
    }
    return metrics


def print_metrics_table(metrics_list: List[Dict[str, Any]]) -> None:
    """Print a comparison table for multiple models."""
    header = f"{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12} {'Total Cost':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(
            f"{m['model']:<25} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} "
            f"{m['weighted_f1']:>12.4f} {m['total_cost']:>12.1f}"
        )
    print("=" * len(header) + "\n")


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    model_name: str = "Final Model",
    save_path: Path = CONFUSION_MATRIX_PATH
) -> None:
    """Plot and save a styled confusion matrix."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1A202C")
    ax.set_facecolor("#1A202C")

    # Draw cells
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "#2D3748" if i != j else "#276749"
            rect = mpatches.FancyBboxPatch(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="#4A5568", linewidth=0.8
            )
            ax.add_patch(rect)
            count = cm[i, j]
            pct = cm_pct[i, j]
            ax.text(j, i - 0.07, str(count), ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")
            ax.text(j, i + 0.18, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=8, color="#A0AEC0")

    # Class label patches on axes
    for idx, cls in enumerate(CLASSES):
        color = CLASS_COLORS[cls]
        for pos, is_x in [(idx, True), (idx, False)]:
            ax.text(
                idx if is_x else -0.65,
                -0.75 if is_x else idx,
                cls, ha="center", va="center",
                fontsize=9, fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2D3748", edgecolor=color, linewidth=1)
            )

    ax.set_xlim(-1.0, len(CLASSES) - 0.5)
    ax.set_ylim(len(CLASSES) - 0.5, -1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Predicted Class", color="#CBD5E0", labelpad=30)
    ax.set_ylabel("Actual Class", color="#CBD5E0", labelpad=50)
    ax.set_title(f"Confusion Matrix - {model_name}", color="white",
                 fontsize=13, fontweight="bold", pad=15)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_metrics(metrics: Dict[str, Any], path: Path = METRICS_PATH) -> None:
    """Save metrics dict to a JSON file."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")
