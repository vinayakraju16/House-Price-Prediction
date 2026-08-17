import uuid
from pathlib import Path
from unittest.mock import patch

from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase, TestCase, override_settings

from djanoproject.database import database_settings

from .models import PredictionLog
from .services import FEATURE_NAMES, engineer_features_for_inference, estimate_details


class DatabaseConfigurationTests(SimpleTestCase):
    def test_sqlite_is_the_default(self):
        config = database_settings({}, base_dir=Path("/application"))

        self.assertEqual(config["ENGINE"], "django.db.backends.sqlite3")
        self.assertEqual(config["NAME"], Path("/application/db.sqlite3"))

    def test_postgresql_configuration_is_explicit_and_health_checked(self):
        config = database_settings(
            {
                "DJANGO_DB_ENGINE": "postgresql",
                "POSTGRES_DB": "house_price",
                "POSTGRES_USER": "app_user",
                "POSTGRES_PASSWORD": "not-logged",
                "POSTGRES_HOST": "database.internal",
                "POSTGRES_PORT": "5433",
                "POSTGRES_CONN_MAX_AGE": "120",
                "POSTGRES_SSLMODE": "require",
            }
        )

        self.assertEqual(config["ENGINE"], "django.db.backends.postgresql")
        self.assertEqual(config["PORT"], 5433)
        self.assertEqual(config["CONN_MAX_AGE"], 120)
        self.assertTrue(config["CONN_HEALTH_CHECKS"])
        self.assertEqual(config["OPTIONS"], {"sslmode": "require"})

    def test_postgresql_requires_credentials_and_host(self):
        with self.assertRaisesMessage(ImproperlyConfigured, "POSTGRES_PASSWORD"):
            database_settings(
                {
                    "DJANGO_DB_ENGINE": "postgresql",
                    "POSTGRES_DB": "house_price",
                    "POSTGRES_USER": "app_user",
                    "POSTGRES_HOST": "database.internal",
                }
            )

    def test_invalid_database_engine_is_rejected(self):
        with self.assertRaisesMessage(ImproperlyConfigured, "DJANGO_DB_ENGINE"):
            database_settings({"DJANGO_DB_ENGINE": "mysql"})

    def test_invalid_postgresql_port_is_rejected(self):
        with self.assertRaisesMessage(ImproperlyConfigured, "POSTGRES_PORT"):
            database_settings(
                {
                    "DJANGO_DB_ENGINE": "postgresql",
                    "POSTGRES_DB": "house_price",
                    "POSTGRES_USER": "app_user",
                    "POSTGRES_PASSWORD": "not-logged",
                    "POSTGRES_HOST": "database.internal",
                    "POSTGRES_PORT": "not-a-port",
                }
            )

VALID_PAYLOAD = {
    "beds": 3,
    "baths": 2.5,
    "size": 2590,
    "lot_size": 6000,
    "zip_code": 98144,
}


class PredictApiTests(TestCase):
    @patch("api.views.estimate_details", return_value={
        "prediction": 234567.891,
        "range": {"low": 196491.59, "high": 272644.19, "margin": 38076.30,
                  "method": "OOF residual quantiles", "nominal_coverage": 0.9},
        "confidence": {"level": "High", "message": "Typical", "unusual_features": []},
        "model_version": "seattle-ridge-v1",
        "model_factors": [{"feature": "numerical__size", "abs_importance": 430003.4}],
        "explanation": {"method": "exact Shapley", "base_value": 200000.0,
                        "top_factors": [{"feature": "size", "contribution": 34567.891}]},
        "comparables": [{"price": 230000.0, "zip_code": 98144, "similarity": 0.9}],
    })
    def test_returns_prediction_range_and_confidence(self, mock_estimate):
        response = self.client.post("/predict/", VALID_PAYLOAD, content_type="application/json")

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["prediction"], 234567.89)
        self.assertEqual(result["range"]["low"], 196491.59)
        self.assertEqual(result["range"]["nominal_coverage"], 0.9)
        self.assertEqual(result["confidence"]["level"], "High")
        self.assertEqual(result["model_version"], "seattle-ridge-v1")
        self.assertEqual(result["top_factors"][0]["feature"], "numerical__size")
        self.assertEqual(result["explanation"]["method"], "exact Shapley")
        self.assertEqual(result["comparables"][0]["zip_code"], 98144)
        self.assertIn("not a formal appraisal", result["disclaimer"])
        uuid.UUID(result["request_id"])
        self.assertEqual(response.headers["X-Request-ID"], result["request_id"])
        event = PredictionLog.objects.get()
        self.assertEqual(event.model_version, "seattle-ridge-v1")
        self.assertEqual(event.status, "success")
        self.assertGreaterEqual(event.latency_ms, 0)
        mock_estimate.assert_called_once_with(VALID_PAYLOAD)

    def test_rejects_missing_features(self):
        payload = dict(VALID_PAYLOAD)
        payload.pop(FEATURE_NAMES[0])
        response = self.client.post("/predict/", payload, content_type="application/json")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "validation_error")
        self.assertEqual(response.json()["details"][FEATURE_NAMES[0]], "This field is required.")

    def test_rejects_malformed_json(self):
        response = self.client.post("/predict/", "not json", content_type="application/json")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "malformed_json")
        self.assertIn("valid JSON", response.json()["details"]["body"])
        self.assertEqual(PredictionLog.objects.get().error_code, "malformed_json")

    def test_rejects_fractional_bedrooms(self):
        response = self.client.post(
            "/predict/", {**VALID_PAYLOAD, "beds": 2.5}, content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["details"]["beds"], "Must be a whole number.")

    def test_rejects_zero_size(self):
        response = self.client.post(
            "/predict/", {**VALID_PAYLOAD, "size": 0}, content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["details"]["size"], "Must be greater than zero.")

    def test_rejects_negative_size_with_field_error(self):
        response = self.client.post(
            "/predict/", {**VALID_PAYLOAD, "size": -1}, content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "validation_error")
        self.assertEqual(response.json()["details"], {"size": "Must be greater than zero."})

    def test_rejects_invalid_zip_with_field_error(self):
        response = self.client.post(
            "/predict/", {**VALID_PAYLOAD, "zip_code": "9814A"}, content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["details"], {"zip_code": "Must be a valid 5-digit ZIP code."}
        )

    def test_rejects_get_requests(self):
        self.assertEqual(self.client.get("/predict/").status_code, 405)

    @patch("api.views.estimate_details", side_effect=RuntimeError("artifact missing"))
    def test_returns_service_unavailable_when_model_cannot_load(self, _mock_estimate):
        response = self.client.post("/predict/", VALID_PAYLOAD, content_type="application/json")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "Prediction service is unavailable.")
        self.assertEqual(PredictionLog.objects.get().error_code, "model_unavailable")

    @patch("api.views.model_metadata", return_value={
        "deployed_model": "gradient_boost", "dataset_rows": 1460,
        "training_rows": 1168, "test_rows": 292,
        "metrics": {"rmse": 38076.3, "r2": 0.811},
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

    @patch("api.views.market_context", return_value={"available": True, "pincode": "77002", "county": "Harris County"})
    @patch("api.views.estimate_details", return_value={
        "prediction": 245000.0,
        "range": {"low": 206000.0, "high": 284000.0, "margin": 39000.0},
        "confidence": {"level": "High", "message": "Typical", "unusual_features": []},
        "model_version": "seattle-ridge-v1",
        "model_factors": [],
    })
    def test_uses_pincode_for_market_context(self, mock_estimate, mock_market_context):
        response = self.client.post("/predict/", {**VALID_PAYLOAD, "pincode": "77002"}, content_type="application/json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["market"]["pincode"], "77002")
        mock_market_context.assert_called_once_with("77002")

    def test_rejects_model_with_target_derived_features(self):
        import pandas as pd

        frame = pd.DataFrame([{name: VALID_PAYLOAD[name] for name in FEATURE_NAMES}])
        with self.assertRaisesRegex(RuntimeError, "target-derived features"):
            engineer_features_for_inference(frame, feature_names=[*FEATURE_NAMES, "price_per_sqft"])

    def test_deployed_estimate_has_additive_explanation_interval_and_comparables(self):
        estimate = estimate_details(VALID_PAYLOAD)

        self.assertEqual(estimate["model_version"], "3.0.0")
        self.assertIn("temporal-holdout residual quantiles", estimate["range"]["method"])
        self.assertAlmostEqual(estimate["range"]["nominal_coverage"], 0.9)
        explanation = estimate["explanation"]
        explained_prediction = explanation["base_value"] + sum(
            factor["contribution"] for factor in explanation["top_factors"]
        )
        self.assertAlmostEqual(explained_prediction, estimate["prediction"], places=5)
        self.assertEqual(len(explanation["top_factors"]), 18)
        self.assertEqual(len(estimate["comparables"]), 5)
        self.assertTrue(all(
            item["source"] == "King County Assessor historical sale"
            for item in estimate["comparables"]
        ))

    def test_accepts_rich_optional_property_fields(self):
        estimate = estimate_details({
            **VALID_PAYLOAD,
            "stories": 2,
            "grade": 8,
            "condition": 4,
            "year_built": 1995,
            "finished_basement_sqft": 400,
            "garage_sqft": 350,
            "fireplaces": 1,
            "has_view": 0,
        })

        self.assertGreater(estimate["prediction"], 0)


class HealthApiTests(TestCase):
    def test_liveness_does_not_require_model(self):
        response = self.client.get("/health/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_readiness_checks_database_and_model(self):
        response = self.client.get("/ready/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], "3.0.0")

    @patch("api.views.load_models", side_effect=RuntimeError("artifact unavailable"))
    def test_readiness_reports_dependency_failure(self, _mock_load):
        response = self.client.get("/ready/")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "unavailable"})


@override_settings(MONITORING_API_KEY="")
class MonitoringApiTests(TestCase):
    def setUp(self):
        PredictionLog.objects.create(
            endpoint="predict", status="success", model_version="2.2.0", latency_ms=12.5,
            input_features=VALID_PAYLOAD, prediction=500000, range_low=400000,
            range_high=600000, confidence_level="High",
        )
        PredictionLog.objects.create(
            endpoint="predict", status="error", error_code="validation_error",
            model_version="2.2.0", latency_ms=2.5,
        )

    def test_returns_privacy_safe_operational_summary(self):
        response = self.client.get("/monitoring/?days=7")

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["requests"], {
            "total": 2, "successful": 1, "errors": 1, "error_rate": 0.5,
        })
        self.assertEqual(result["latency_ms"]["mean"], 7.5)
        self.assertEqual(result["errors_by_code"]["validation_error"], 1)
        self.assertEqual(result["feature_monitoring"]["beds"]["observed_count"], 1)
        self.assertNotIn("input_features", result["recent"][0])

    @override_settings(MONITORING_API_KEY="secret-monitoring-key")
    def test_monitoring_key_is_required_when_configured(self):
        self.assertEqual(self.client.get("/monitoring/").status_code, 403)
        response = self.client.get("/monitoring/", HTTP_X_MONITORING_KEY="secret-monitoring-key")
        self.assertEqual(response.status_code, 200)

    def test_rejects_invalid_monitoring_window(self):
        response = self.client.get("/monitoring/?days=weekly")
        self.assertEqual(response.status_code, 400)
