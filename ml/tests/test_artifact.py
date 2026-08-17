import hashlib
import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class ActiveArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_dir = BACKEND_DIR / "mlmodels"
        cls.registry = json.loads((model_dir / "model_registry.json").read_text(encoding="utf-8"))
        cls.version = cls.registry["active_version"]
        cls.entry = cls.registry["models"][cls.version]
        cls.bundle = joblib.load(model_dir / cls.entry["artifact"])
        cls.metadata = json.loads((model_dir / cls.entry["metadata"]).read_text(encoding="utf-8"))

    def test_registry_bundle_and_metadata_versions_match(self):
        self.assertEqual(self.bundle["model_version"], self.version)
        self.assertEqual(self.metadata["model_version"], self.version)

    def test_registry_checksums_match_versioned_files(self):
        model_dir = BACKEND_DIR / "mlmodels"
        for registry_key, path_key in (("artifact_sha256", "artifact"), ("metadata_sha256", "metadata")):
            digest = hashlib.sha256((model_dir / self.entry[path_key]).read_bytes()).hexdigest()
            self.assertEqual(self.entry[registry_key], digest)
        self.assertEqual(self.metadata["artifact"]["sha256"], self.entry["artifact_sha256"])
        self.assertEqual(self.metadata["metadata_schema_version"], "2.0")

    def test_artifact_runtime_matches_installed_sklearn(self):
        self.assertEqual(self.metadata["runtime"]["scikit_learn"], sklearn.__version__)

    def test_artifact_uses_only_inference_available_features(self):
        self.assertEqual(self.bundle["feature_names"], self.metadata["features"])
        forbidden = {"price", "price_per_sqft", "is_price_anomaly"}
        self.assertFalse(forbidden.intersection(self.bundle["feature_names"]))
        self.assertEqual(
            set(self.bundle["required_features"]),
            {"beds", "baths", "size", "lot_size", "zip_code"},
        )

    def test_prediction_is_finite_and_positive(self):
        values = dict(self.metadata["feature_defaults"])
        values.update({
            "beds": 3, "baths": 2.5, "size": 1800, "lot_size": 5000,
            "zip_code": "98144",
        })
        features = pd.DataFrame([values], columns=self.bundle["feature_names"])
        prediction = self.bundle["best_model"].predict(features)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())
