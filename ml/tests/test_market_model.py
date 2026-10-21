import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from ml.features.build_features import build_property_market_preprocessor
from model_runtime.market import MarketRelativeRegressor, PropertyMarketFeatureEngineer

NUMERIC = ["size", "lot_size", "year_built", "year_renovated", "sale_year"]
CATEGORICAL = ["zip_code", "heat_system"]
MARKET = ["zhvi"]


def _frame(rows=60, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "size": rng.integers(800, 4000, rows).astype(float),
        "lot_size": rng.integers(2000, 20000, rows).astype(float),
        "year_built": rng.integers(1950, 2026, rows).astype(float),
        "year_renovated": np.where(rng.random(rows) < 0.2, 2015.0, np.nan),
        "sale_year": 2026.0,
        "zip_code": rng.choice(["27615", "27587"], rows),
        "heat_system": pd.array(rng.choice(["A", "K", None], rows), dtype="string"),
        "zhvi": rng.choice([400_000.0, 500_000.0], rows),
    })
    frame["price"] = frame["size"] * 200 * frame["zhvi"] / 450_000
    return frame


class PropertyMarketFeatureTests(unittest.TestCase):
    def test_derived_features_and_missing_categories(self):
        frame = _frame(4)
        frame.loc[0, ["year_built", "sale_year"]] = [2025, 2026]
        frame.loc[1, ["year_built", "lot_size"]] = [1990, 0]
        out = PropertyMarketFeatureEngineer(NUMERIC, CATEGORICAL, MARKET).transform(frame)
        self.assertEqual(out.loc[0, "is_new_construction"], 1.0)
        self.assertEqual(out.loc[1, "is_new_construction"], 0.0)
        self.assertTrue(np.isnan(out.loc[1, "living_lot_ratio"]))
        self.assertTrue(all(isinstance(v, str) or v != v for v in out["heat_system"]))

    def test_pipeline_fits_with_missing_categorical_values(self):
        frame = _frame()
        pipeline = Pipeline([
            ("preprocessor", build_property_market_preprocessor(NUMERIC, CATEGORICAL, MARKET)),
            ("regressor", HistGradientBoostingRegressor(max_iter=20)),
        ])
        pipeline.fit(frame.drop(columns="price"), np.log(frame["price"]))
        self.assertEqual(len(pipeline.predict(frame.drop(columns="price"))), len(frame))


class MarketRelativeRegressorTests(unittest.TestCase):
    def test_estimate_follows_the_index_without_refitting(self):
        frame = _frame(200)
        features = frame.drop(columns="price")
        model = MarketRelativeRegressor(Pipeline([
            ("preprocessor", build_property_market_preprocessor(NUMERIC, CATEGORICAL, [])),
            ("regressor", HistGradientBoostingRegressor(max_iter=50)),
        ])).fit(features, frame["price"])
        row = features.iloc[[0]].copy()
        before = model.predict(row)[0]
        row["zhvi"] = row["zhvi"] * 1.10
        self.assertAlmostEqual(model.predict(row)[0] / before, 1.10, places=6)

    def test_missing_index_falls_back_to_zip_median(self):
        frame = _frame(200)
        features = frame.drop(columns="price")
        model = MarketRelativeRegressor(Pipeline([
            ("preprocessor", build_property_market_preprocessor(NUMERIC, CATEGORICAL, [])),
            ("regressor", HistGradientBoostingRegressor(max_iter=20)),
        ])).fit(features, frame["price"])
        row = features.iloc[[0]].copy()
        row["zhvi"] = np.nan
        self.assertTrue(np.isfinite(model.predict(row)[0]))


if __name__ == "__main__":
    unittest.main()
