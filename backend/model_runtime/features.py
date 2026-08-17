"""Inference-safe Seattle feature transformations.

This module lives under ``backend`` so joblib can import the transformer when Django
loads an artifact. Offline training adds ``backend`` to ``sys.path`` and imports the
same module name, keeping the serialized reference stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_FEATURES = ("beds", "baths", "size", "lot_size", "zip_code")
KING_COUNTY_FEATURES = (
    "beds", "baths", "size", "lot_size", "zip_code", "stories", "grade",
    "condition", "year_built", "year_renovated", "finished_basement_sqft",
    "garage_sqft", "fireplaces", "heat_system", "has_view", "property_type",
    "sale_year", "sale_month",
)
KING_COUNTY_NUMERIC_FEATURES = (
    "beds", "baths", "size", "lot_size", "stories", "grade", "condition",
    "year_built", "year_renovated", "finished_basement_sqft", "garage_sqft",
    "fireplaces", "has_view", "sale_year", "sale_month", "property_age",
    "years_since_renovation", "sqft_per_bedroom", "bathroom_bedroom_ratio",
    "living_lot_ratio", "total_rooms", "was_renovated",
)
KING_COUNTY_CATEGORICAL_FEATURES = ("zip_code", "heat_system", "property_type")
NUMERIC_FEATURES = (
    "beds",
    "baths",
    "size",
    "lot_size",
    "has_lot_data",
    "lot_to_house_ratio",
    "beds_plus_baths",
    "sqft_per_bedroom",
    "bathroom_bedroom_ratio",
    "is_small_unit",
)
CATEGORICAL_FEATURES = ("zip_code", "size_category")


class SeattleFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive only features available from the five API request fields."""

    def fit(self, features, target=None):
        self._validate_columns(features)
        return self

    def transform(self, features):
        self._validate_columns(features)
        engineered = features.loc[:, BASE_FEATURES].copy()
        for column in BASE_FEATURES:
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce")

        size = engineered["size"].to_numpy(dtype=float)
        beds = engineered["beds"].to_numpy(dtype=float)
        lot_size = engineered["lot_size"].to_numpy(dtype=float)
        baths = engineered["baths"].to_numpy(dtype=float)

        engineered["has_lot_data"] = (engineered["lot_size"] > 0).astype(int)
        engineered["lot_to_house_ratio"] = np.divide(
            lot_size, size, out=np.zeros_like(lot_size), where=size > 0
        ).clip(0, 100)
        engineered["beds_plus_baths"] = engineered["beds"] + engineered["baths"]
        engineered["sqft_per_bedroom"] = np.divide(
            size, beds, out=np.zeros_like(size), where=beds > 0
        ).clip(0, 10_000)
        engineered["bathroom_bedroom_ratio"] = np.divide(
            baths, beds, out=np.zeros_like(baths), where=beds > 0
        ).clip(0, 10)
        engineered["is_small_unit"] = ((engineered["beds"] <= 2) & (engineered["size"] < 1200)).astype(int)
        engineered["size_category"] = pd.cut(
            engineered["size"],
            bins=[-np.inf, 1000, 2000, 3500, np.inf],
            labels=["small", "medium", "large", "xlarge"],
        ).astype(object)
        return engineered

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*NUMERIC_FEATURES, *CATEGORICAL_FEATURES], dtype=object)

    @staticmethod
    def _validate_columns(features):
        if not isinstance(features, pd.DataFrame):
            raise TypeError("SeattleFeatureEngineer requires a pandas DataFrame")
        missing = [column for column in BASE_FEATURES if column not in features.columns]
        if missing:
            raise ValueError(f"Missing base features: {', '.join(missing)}")


class KingCountyFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create inference-safe features available in the official county extracts."""

    def fit(self, features, target=None):
        self._validate_columns(features)
        return self

    def transform(self, features):
        self._validate_columns(features)
        result = features.loc[:, KING_COUNTY_FEATURES].copy()
        categorical = set(KING_COUNTY_CATEGORICAL_FEATURES)
        for column in KING_COUNTY_FEATURES:
            if column not in categorical:
                result[column] = pd.to_numeric(result[column], errors="coerce")
        for column in categorical:
            result[column] = result[column].astype("string")

        size = result["size"].to_numpy(dtype=float)
        beds = result["beds"].to_numpy(dtype=float)
        baths = result["baths"].to_numpy(dtype=float)
        lot = result["lot_size"].to_numpy(dtype=float)
        built = result["year_built"].to_numpy(dtype=float)
        renovated = result["year_renovated"].to_numpy(dtype=float)
        sale_year = result["sale_year"].to_numpy(dtype=float)
        result["property_age"] = np.maximum(sale_year - built, 0)
        result["was_renovated"] = np.isfinite(renovated).astype(int)
        result["years_since_renovation"] = np.where(
            np.isfinite(renovated), np.maximum(sale_year - renovated, 0), np.nan
        )
        result["sqft_per_bedroom"] = np.divide(
            size, beds, out=np.full_like(size, np.nan), where=beds > 0
        )
        result["bathroom_bedroom_ratio"] = np.divide(
            baths, beds, out=np.full_like(baths, np.nan), where=beds > 0
        )
        result["living_lot_ratio"] = np.divide(
            size, lot, out=np.full_like(size, np.nan), where=lot > 0
        ).clip(0, 20)
        result["total_rooms"] = beds + baths
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            [*KING_COUNTY_NUMERIC_FEATURES, *KING_COUNTY_CATEGORICAL_FEATURES],
            dtype=object,
        )

    @staticmethod
    def _validate_columns(features):
        if not isinstance(features, pd.DataFrame):
            raise TypeError("KingCountyFeatureEngineer requires a pandas DataFrame")
        missing = [column for column in KING_COUNTY_FEATURES if column not in features.columns]
        if missing:
            raise ValueError(f"Missing King County features: {', '.join(missing)}")
