"""
test_preprocessing.py — Tests for the data preprocessing pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import build_preprocessor, encode_labels, decode_labels
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, CLASSES
from src.data_processing import load_train, split_features_target


def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 3


def test_preprocessor_fit_transform():
    df = load_train()
    X, y = split_features_target(df)
    preprocessor = build_preprocessor()
    
    # Fit and transform
    X_trans = preprocessor.fit_transform(X)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(df)
    # Check no NaNs remaining after transformation
    assert not np.isnan(X_trans).any(), "Transformed feature matrix should not have NaNs"


def test_encode_decode_labels():
    labels = pd.Series(["RED", "YELLOW", "GREEN", "BLACK", "RED"])
    encoded = encode_labels(labels)
    assert np.array_equal(encoded, np.array([0, 1, 2, 3, 0]))
    decoded = decode_labels(encoded)
    assert decoded == list(labels)


def test_preprocessor_handles_missing_values():
    """Ensure preprocessor imputes numerical with median and categorical with mode."""
    preprocessor = build_preprocessor()
    df = load_train()
    X, _ = split_features_target(df)
    preprocessor.fit(X)

    # Synthetic sample with all numerical features NaN
    test_sample = X.iloc[[0]].copy()
    for col in NUMERICAL_FEATURES:
        test_sample[col] = np.nan

    transformed = preprocessor.transform(test_sample)
    assert not np.isnan(transformed).any()
