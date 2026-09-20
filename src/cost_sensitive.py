"""
cost_sensitive.py — Cost-sensitive decision layer for triage classification.

The competition supplies an unequal cost matrix. Instead of naively taking
argmax(probabilities), this module computes the EXPECTED COST of each
possible predicted class and selects the class with the MINIMUM expected cost.

Expected Cost formula:
    EC(predicted_class) = SUM_over_actual_classes [ P(actual_class) * Cost(actual, predicted) ]

The cost matrix is taken EXACTLY from the competition problem statement.
DO NOT modify COST_MATRIX here — edit src/config.py only.
"""
import numpy as np
from typing import Dict, List, Tuple

from src.config import COST_MATRIX, CLASSES


# Precompute as numpy array for vectorised operations
_COST_ARRAY = np.array(COST_MATRIX, dtype=float)  # shape (4, 4)


def compute_expected_costs(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute the expected cost for each possible predicted class.

    Args:
        probabilities: 1-D array of shape (4,) with P(RED), P(YELLOW), P(GREEN), P(BLACK).
                       Must sum to approximately 1.

    Returns:
        1-D array of shape (4,) — expected cost for predicting each class.

    Math:
        EC[j] = sum_i ( P[i] * Cost[i, j] )
        Vectorised: EC = probabilities @ _COST_ARRAY
    """
    if probabilities.ndim != 1 or len(probabilities) != 4:
        raise ValueError(f"Expected 1-D probability array of length 4, got shape {probabilities.shape}")
    return probabilities @ _COST_ARRAY  # shape (4,)


def compute_expected_costs_batch(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Compute expected costs for a batch of patients.

    Args:
        prob_matrix: 2-D array of shape (n_patients, 4).

    Returns:
        2-D array of shape (n_patients, 4) — expected costs per patient per predicted class.
    """
    if prob_matrix.ndim != 2 or prob_matrix.shape[1] != 4:
        raise ValueError(f"Expected 2-D array of shape (n, 4), got shape {prob_matrix.shape}")
    return prob_matrix @ _COST_ARRAY


def predict_cost_sensitive(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Select the predicted class with the minimum expected cost for each patient.

    Args:
        prob_matrix: 2-D array of shape (n_patients, 4).

    Returns:
        1-D integer array of shape (n_patients,) — predicted class indices.
    """
    expected_costs = compute_expected_costs_batch(prob_matrix)
    return np.argmin(expected_costs, axis=1)


def predict_cost_sensitive_labels(prob_matrix: np.ndarray) -> List[str]:
    """
    Return predicted class labels (strings) for each patient.

    Args:
        prob_matrix: 2-D array of shape (n_patients, 4).

    Returns:
        List of predicted triage labels (e.g., ['RED', 'GREEN', ...]).
    """
    indices = predict_cost_sensitive(prob_matrix)
    return [CLASSES[i] for i in indices]


def get_expected_costs_dict(probabilities: np.ndarray) -> Dict[str, float]:
    """
    Return a human-readable dict of expected costs for a single patient.

    Args:
        probabilities: 1-D array of shape (4,).

    Returns:
        {'RED': ..., 'YELLOW': ..., 'GREEN': ..., 'BLACK': ...}
    """
    costs = compute_expected_costs(probabilities)
    return {cls: round(float(costs[i]), 4) for i, cls in enumerate(CLASSES)}


def get_cost_matrix_as_dict() -> Dict[str, Dict[str, int]]:
    """Return the cost matrix as a nested dict for API responses."""
    return {
        actual: {predicted: COST_MATRIX[i][j] for j, predicted in enumerate(CLASSES)}
        for i, actual in enumerate(CLASSES)
    }


def total_misclassification_cost(
    y_true: List[str],
    y_pred: List[str]
) -> float:
    """
    Compute total misclassification cost over a dataset.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Total cost (float).
    """
    total = 0.0
    from src.config import CLASS_INDEX
    for actual, predicted in zip(y_true, y_pred):
        i = CLASS_INDEX[actual]
        j = CLASS_INDEX[predicted]
        total += COST_MATRIX[i][j]
    return total
