from unittest.mock import patch

from django.test import TestCase

from .services import FEATURE_NAMES


VALID_PAYLOAD = {
    "MasVnrArea": 100,
    "SaleType_WD": 1,
    "OverallQual": 7,
    "OverallCond": 5,
    "ExterQual_Gd": 1,
    "ExterCond_Fa": 0,
    "BsmtUnfSF": 300,
    "BsmtFinType1_LwQ": 0,
    "LotArea": 9600,
    "YearBuilt": 2005,
    "BsmtFinSF1": 500,
    "TotRmsAbvGrd": 7,
    "GarageCars": 2,
    "GarageArea": 480,
}


class PredictApiTests(TestCase):
    @patch("api.views.estimate_details", return_value={
        "prediction": 234567.891,
        "range": {"low": 196491.59, "high": 272644.19, "margin": 38076.30},
        "confidence": {"level": "High", "message": "Typical", "unusual_features": []},
    })
    def test_returns_prediction_range_and_confidence(self, mock_estimate):
        response = self.client.post("/predict/", VALID_PAYLOAD, content_type="application/json")

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["prediction"], 234567.89)
        self.assertEqual(result["range"]["low"], 196491.59)
        self.assertEqual(result["confidence"]["level"], "High")
        self.assertIn("not a formal appraisal", result["disclaimer"])
        mock_estimate.assert_called_once_with(VALID_PAYLOAD)

    def test_rejects_missing_features(self):
        payload = dict(VALID_PAYLOAD)
        payload.pop(FEATURE_NAMES[0])
        response = self.client.post("/predict/", payload, content_type="application/json")

        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing features", response.json()["error"])

    def test_rejects_malformed_json(self):
        response = self.client.post("/predict/", "not json", content_type="application/json")

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid JSON", response.json()["error"])

    def test_rejects_get_requests(self):
        self.assertEqual(self.client.get("/predict/").status_code, 405)

    @patch("api.views.model_metadata", return_value={
        "deployed_model": "gradient_boost", "dataset_rows": 1460,
        "training_rows": 1168, "test_rows": 292,
        "gradient_boost": {"rmse": 38076.3, "r2": 0.811},
        "target_summary": {"minimum": 34900, "median": 163000, "maximum": 755000},
        "feature_importance": [{"feature": "OverallQual", "importance": 0.5}],
    })
    def test_returns_model_information(self, mock_metadata):
        response = self.client.get("/model-info/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["metrics"]["r2"], 0.811)
        self.assertEqual(response.json()["feature_importance"][0]["feature"], "OverallQual")
        mock_metadata.assert_called_once()

    @patch("api.views.harris_estimate", return_value={
        "prediction": 300000.0,
        "range": {"low": 274000.0, "high": 326000.0, "margin": 26000.0},
        "confidence": {"level": "Medium", "message": "Test"},
        "disclaimer": "HCAD appraisal market value, not verified sale price.",
    })
    def test_returns_harris_estimate(self, mock_estimate):
        response = self.client.post("/predict-harris/", {"sample": "payload"}, content_type="application/json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], 300000.0)
        self.assertEqual(response.json()["confidence"]["level"], "Medium")
        mock_estimate.assert_called_once_with({"sample": "payload"})
