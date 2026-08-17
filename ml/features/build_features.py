"""Construct leakage-safe preprocessing for Seattle candidate models."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_runtime.features import (  # noqa: E402
    BASE_FEATURES,
    CATEGORICAL_FEATURES,
    KING_COUNTY_CATEGORICAL_FEATURES,
    KING_COUNTY_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    KingCountyFeatureEngineer,
    SeattleFeatureEngineer,
)


def _numeric_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _categorical_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_base_preprocessor() -> ColumnTransformer:
    """Preprocess the five raw request fields, treating ZIP as categorical."""
    return ColumnTransformer([
        ("numeric", _numeric_pipeline(), list(BASE_FEATURES[:-1])),
        ("categorical", _categorical_pipeline(), ["zip_code"]),
    ])


def build_engineered_preprocessor() -> Pipeline:
    """Create safe derived features and preprocess them within the fitted pipeline."""
    return Pipeline([
        ("features", SeattleFeatureEngineer()),
        ("columns", ColumnTransformer([
            ("numeric", _numeric_pipeline(), list(NUMERIC_FEATURES)),
            ("categorical", _categorical_pipeline(), list(CATEGORICAL_FEATURES)),
        ])),
    ])


def build_king_county_preprocessor() -> Pipeline:
    """Build the shared rich-feature transformer used by training and inference."""
    return Pipeline([
        ("features", KingCountyFeatureEngineer()),
        ("columns", ColumnTransformer([
            ("numeric", _numeric_pipeline(), list(KING_COUNTY_NUMERIC_FEATURES)),
            ("categorical", _categorical_pipeline(), list(KING_COUNTY_CATEGORICAL_FEATURES)),
        ])),
    ])
