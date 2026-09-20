"""
test_cost_matrix.py — Tests for the cost-sensitive decision layer and competition cost matrix.
"""
import pytest
import numpy as np
from src.config import COST_MATRIX, CLASSES, CLASS_INDEX
from src.cost_sensitive import (
    compute_expected_costs,
    compute_expected_costs_batch,
    predict_cost_sensitive,
    predict_cost_sensitive_labels,
    total_misclassification_cost,
    get_expected_costs_dict
)


def test_cost_matrix_dimensions_and_diagonal():
    """Verify cost matrix matches competition spec."""
    cm = np.array(COST_MATRIX)
    assert cm.shape == (4, 4), "Cost matrix must be 4x4"
    assert np.all(np.diag(cm) == 0), "Diagonal must be 0 (zero cost for correct prediction)"


def test_cost_matrix_exact_values():
    """Verify exact values from MM26ML03 problem statement."""
    # RED row: [0, 5, 10, 5]
    assert COST_MATRIX[0] == [0, 5, 10, 5]
    # YELLOW row: [2, 0, 3, 2]
    assert COST_MATRIX[1] == [2, 0, 3, 2]
    # GREEN row: [1, 1, 0, 1]
    assert COST_MATRIX[2] == [1, 1, 0, 1]
    # BLACK row: [2, 2, 2, 0]
    assert COST_MATRIX[3] == [2, 2, 2, 0]


def test_pure_probability_expected_costs():
    """If P(RED) = 1.0, expected cost for predicting each class should be row 0."""
    p_red = np.array([1.0, 0.0, 0.0, 0.0])
    costs = compute_expected_costs(p_red)
    np.testing.assert_allclose(costs, [0, 5, 10, 5])

    # Minimum cost prediction should be RED (index 0)
    assert np.argmin(costs) == 0


def test_cost_sensitive_bias_towards_red():
    """
    Given a scenario where patient is 50% RED and 50% YELLOW:
    Predicting RED: 0.5*0 + 0.5*2 = 1.0
    Predicting YELLOW: 0.5*5 + 0.5*0 = 2.5
    Predicting GREEN: 0.5*10 + 0.5*3 = 6.5
    Predicting BLACK: 0.5*5 + 0.5*2 = 3.5
    The cost-sensitive decision should choose RED because misclassifying RED as YELLOW costs 5,
    while misclassifying YELLOW as RED costs only 2!
    """
    prob = np.array([0.5, 0.5, 0.0, 0.0])
    costs = compute_expected_costs(prob)
    assert costs[0] == 1.0
    assert costs[1] == 2.5
    best_class_idx = np.argmin(costs)
    assert best_class_idx == 0  # RED


def test_batch_expected_costs():
    """Test batch matrix multiplication."""
    probs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    labels = predict_cost_sensitive_labels(probs)
    assert labels == ["RED", "YELLOW", "GREEN", "BLACK"]


def test_total_misclassification_cost():
    y_true = ["RED", "YELLOW", "GREEN", "BLACK"]
    y_pred = ["RED", "YELLOW", "GREEN", "BLACK"]
    assert total_misclassification_cost(y_true, y_pred) == 0.0

    # Misclassify RED as GREEN: cost 10
    y_pred_bad = ["GREEN", "YELLOW", "GREEN", "BLACK"]
    assert total_misclassification_cost(y_true, y_pred_bad) == 10.0


def test_get_expected_costs_dict():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    d = get_expected_costs_dict(p)
    assert set(d.keys()) == set(CLASSES)
    for v in d.values():
        assert isinstance(v, float)
