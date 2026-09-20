"""
preprocessing.py — sklearn-compatible preprocessing pipeline.

Builds a ColumnTransformer that:
  - Imputes + scales numerical features
  - Imputes + one-hot-encodes categorical features

IMPORTANT: The pipeline is FITTED ON TRAINING DATA ONLY.
           Test data is transformed using the fitted pipeline.
           This prevents data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURE, CLASSES


def _clean_text_series(x):
    """Ensure text is a 1D iterable of strings without NaNs."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.Series(x).fillna("").astype(str)


def build_preprocessor() -> ColumnTransformer:
    """
    Build an unfitted ColumnTransformer preprocessing pipeline.

    Numerical: median imputation → standard scaling
    Categorical: most_frequent imputation → one-hot encoding
    Text (description): clean NaNs → TF-IDF keyword vectorization
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    text_pipeline = Pipeline([
        ("clean", FunctionTransformer(_clean_text_series, feature_names_out="one-to-one")),
        ("tfidf", TfidfVectorizer(
            max_features=60,
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("text", text_pipeline, TEXT_FEATURE),
        ],
        remainder="drop",  # Drop any unexpected columns (e.g. patient_id)
        verbose_feature_names_out=True,
    )
    return preprocessor


def encode_labels(y: pd.Series) -> np.ndarray:
    """Encode string class labels to integers using the canonical CLASSES order."""
    mapping = {cls: i for i, cls in enumerate(CLASSES)}
    return np.array([mapping[label] for label in y], dtype=int)


def decode_labels(y_encoded: np.ndarray) -> list:
    """Decode integer labels back to class name strings."""
    return [CLASSES[i] for i in y_encoded]
