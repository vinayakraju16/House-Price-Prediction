import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ml.training.experiment_tracking import log_mlflow_run


class ExperimentTrackingTests(unittest.TestCase):
    def test_optional_dependency_has_clear_failure_message(self):
        with patch.dict(sys.modules, {"mlflow": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-mlflow.txt"):
                log_mlflow_run({}, "missing.joblib")

    def test_logs_distinct_cv_and_legacy_metrics(self):
        run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

        class RunContext:
            def __enter__(self):
                return run

            def __exit__(self, *_args):
                return False

        fake_mlflow = SimpleNamespace(
            set_tracking_uri=MagicMock(), set_experiment=MagicMock(),
            start_run=MagicMock(return_value=RunContext()), log_params=MagicMock(),
            log_metrics=MagicMock(), set_tags=MagicMock(), log_artifact=MagicMock(),
            log_dict=MagicMock(),
        )
        metadata = {
            "model_version": "test-version", "model_type": "TestRegressor",
            "target_transform": {"forward": "log1p"}, "feature_count": 5,
            "hyperparameters": {"max_iter": 10},
            "nested_cv": {
                "mae": {"mean": 10.0}, "rmse": {"mean": 20.0}, "r2": {"mean": 0.5},
            },
            "metrics": {"mae": 11.0, "rmse": 21.0, "r2": 0.4},
            "status": "experimental",
            "dataset_version": {"train_sha256": "abc123"},
        }
        with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            result = log_mlflow_run(metadata, "model.joblib", tracking_uri="sqlite:///test.db")

        self.assertEqual(result["run_id"], "run-123")
        self.assertTrue(result["enabled"])
        logged_metrics = fake_mlflow.log_metrics.call_args.args[0]
        self.assertEqual(logged_metrics["nested_cv_mae_mean"], 10.0)
        self.assertEqual(logged_metrics["legacy_test_mae"], 11.0)
        self.assertNotIn("tracking_uri", result)
