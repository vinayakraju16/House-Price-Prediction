"""Inference-safe building blocks for per-market models (Wake County and later markets).

Like ``features.py`` this lives under ``backend`` so joblib can import the classes when
an artifact is loaded; offline training adds ``backend`` to ``sys.path``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

# Derived columns this transformer adds when their inputs are present.
DERIVED_FEATURES = (
    "property_age", "is_new_construction", "was_renovated", "years_since_renovation",
    "living_lot_ratio",
)


class PropertyMarketFeatureEngineer(BaseEstimator, TransformerMixin):
    """Coerce property, calendar and ZIP-market columns and add derived property features.

    ``numeric`` and ``categorical`` list the raw property columns; ``market`` lists the
    numeric ZIP-month market columns (see ``ml.data.market_features``). ``exclude`` names
    input columns used only to derive features (for example ``sale_year`` for property
    age) that are not passed to the model themselves. The transformer is stateless, so it
    is safe inside cross-validation.
    """

    def __init__(self, numeric=(), categorical=(), market=(), exclude=()):
        self.numeric = numeric
        self.categorical = categorical
        self.market = market
        self.exclude = exclude

    def fit(self, features, target=None):
        self._validate_columns(features)
        return self

    def transform(self, features):
        self._validate_columns(features)
        numeric = [*self.numeric, *self.market]
        result = features.loc[:, [*numeric, *self.categorical]].copy()
        for column in numeric:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
        for column in self.categorical:
            # Plain str/NaN objects: pandas' NA marker breaks SimpleImputer's isnan check.
            result[column] = pd.Series(
                [np.nan if pd.isna(value) else str(value) for value in result[column]],
                index=result.index, dtype=object,
            )

        nan = np.full(len(result), np.nan)
        sale_year = result["sale_year"].to_numpy(dtype=float) if "sale_year" in result else nan
        built = result["year_built"].to_numpy(dtype=float) if "year_built" in result else nan
        renovated = (result["year_renovated"].to_numpy(dtype=float)
                     if "year_renovated" in result else nan)
        size = result["size"].to_numpy(dtype=float) if "size" in result else nan
        lot = result["lot_size"].to_numpy(dtype=float) if "lot_size" in result else nan

        age = np.maximum(sale_year - built, 0)
        result["property_age"] = age
        # Builders' inventory sells on different terms (incentives, rate buydowns) from
        # resales; in 2025-2026 new homes often sold below comparable resales.
        result["is_new_construction"] = np.where(np.isfinite(age), (age <= 1).astype(float),
                                                 np.nan)
        result["was_renovated"] = np.isfinite(renovated).astype(float)
        result["years_since_renovation"] = np.where(
            np.isfinite(renovated), np.maximum(sale_year - renovated, 0), np.nan
        )
        result["living_lot_ratio"] = np.divide(
            size, lot, out=np.full_like(size, np.nan), where=lot > 0
        ).clip(0, 20)
        return result

    def numeric_output_columns(self):
        kept = [name for name in self.numeric if name not in set(self.exclude)]
        return [*kept, *DERIVED_FEATURES, *self.market]

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*self.numeric_output_columns(), *self.categorical], dtype=object)

    def _validate_columns(self, features):
        if not isinstance(features, pd.DataFrame):
            raise TypeError("PropertyMarketFeatureEngineer requires a pandas DataFrame")
        required = [*self.numeric, *self.categorical, *self.market]
        missing = [column for column in required if column not in features.columns]
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")


class MarketRelativeRegressor(BaseEstimator, RegressorMixin):
    """Predict log(price / ZIP home value index), then scale back by the current index.

    Training on the ratio removes the market level from the target, so when a ZIP's index
    moves after training the estimate follows it without refitting. Rows without an index
    value fall back to the training median index of their ZIP, then the global median.
    """

    def __init__(self, regressor=None, index_column="zhvi", zip_column="zip_code"):
        self.regressor = regressor
        self.index_column = index_column
        self.zip_column = zip_column

    def _log_index(self, features: pd.DataFrame) -> np.ndarray:
        index = pd.to_numeric(features[self.index_column], errors="coerce").to_numpy(float)
        index = np.where(index > 0, index, np.nan)
        zips = features[self.zip_column].astype(str).str.zfill(5).to_numpy()
        fallback = np.array([self.zip_log_index_.get(z, self.global_log_index_) for z in zips])
        return np.where(np.isfinite(index), np.log(index), fallback)

    def fit(self, features, target):
        target = np.asarray(target, dtype=float)
        index = pd.to_numeric(features[self.index_column], errors="coerce")
        logs = np.log(index.where(index > 0))
        zips = features[self.zip_column].astype(str).str.zfill(5)
        self.zip_log_index_ = logs.groupby(zips).median().dropna().to_dict()
        self.global_log_index_ = float(np.nanmedian(logs))
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(features, np.log(target) - self._log_index(features))
        return self

    def predict(self, features):
        return np.exp(self.regressor_.predict(features) + self._log_index(features))
