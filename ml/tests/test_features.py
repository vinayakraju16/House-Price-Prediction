import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ml.features.build_features import build_engineered_preprocessor

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from model_runtime.features import SeattleFeatureEngineer  # noqa: E402


class SeattleFeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"beds": 3, "baths": 2.5, "size": 1800, "lot_size": 5000, "zip_code": 98144},
            {"beds": 0, "baths": 1, "size": 500, "lot_size": 0, "zip_code": 98101},
        ])

    def test_safe_division_and_missing_lot_features(self):
        transformed = SeattleFeatureEngineer().fit_transform(self.frame)
        self.assertTrue(np.isfinite(transformed.select_dtypes(include="number").to_numpy()).all())
        self.assertEqual(transformed.loc[1, "sqft_per_bedroom"], 0)
        self.assertEqual(transformed.loc[1, "has_lot_data"], 0)

    def test_transform_does_not_mutate_input(self):
        before = self.frame.copy(deep=True)
        SeattleFeatureEngineer().fit_transform(self.frame)
        pd.testing.assert_frame_equal(self.frame, before)

    def test_complete_pipeline_predicts_finite_values(self):
        pipeline = Pipeline([
            ("preprocessor", build_engineered_preprocessor()),
            ("model", Ridge(alpha=10)),
        ])
        target = pd.Series([800000, 350000])
        pipeline.fit(self.frame, target)
        prediction = pipeline.predict(self.frame)
        self.assertTrue(np.isfinite(prediction).all())
