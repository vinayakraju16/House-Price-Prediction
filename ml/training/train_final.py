"""Fit and register the evidence-selected Seattle model.

Requires `python -m ml.training.tune` and `python -m ml.evaluation.error_analysis`
to have completed successfully.

Run from the repository root:
    python -m ml.training.train_final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import django
import joblib
import numpy as np
import pandas as pd
import sklearn

from ml.data.validation import validate_seattle_dataframe
from ml.training.experiment_tracking import log_mlflow_run
from ml.training.tune import build_candidate

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "Seattle" / "train_cleaned.csv"
TEST_PATH = ROOT / "data" / "Seattle" / "test_cleaned.csv"
TUNING_PATH = ROOT / "reports" / "tuning_results.json"
EVALUATION_PATH = ROOT / "reports" / "model_evaluation.json"
MODEL_DIR = ROOT / "backend" / "mlmodels"
REGISTRY_PATH = MODEL_DIR / "model_registry.json"
COMPATIBILITY_MODEL_PATH = MODEL_DIR / "trained_models.pkl"
COMPATIBILITY_METADATA_PATH = MODEL_DIR / "metadata.json"
MODEL_VERSION = "2.2.0"
FEATURES = ["beds", "baths", "size", "lot_size", "zip_code"]
TARGET = "price"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _typical_ranges(features: pd.DataFrame) -> dict:
    return {
        column: {
            "minimum": float(features[column].min()),
            "p05": float(features[column].quantile(0.05)),
            "p95": float(features[column].quantile(0.95)),
            "maximum": float(features[column].max()),
        }
        for column in FEATURES
    }


def main(track_mlflow=False, tracking_uri=None, experiment_name="house-price-seattle") -> None:
    tuning = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    validate_seattle_dataframe(train, stage="cleaned")
    validate_seattle_dataframe(test, stage="cleaned")

    model = build_candidate()
    model.set_params(**tuning["best_params"])
    model.fit(train[FEATURES], train[TARGET])
    smoke_prediction = model.predict(test[FEATURES].head(5))
    if not np.isfinite(smoke_prediction).all() or not (smoke_prediction > 0).all():
        raise RuntimeError("Final model failed finite-positive prediction validation")

    version_dir = MODEL_DIR / "versions" / MODEL_VERSION
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.joblib"
    metadata_path = version_dir / "metadata.json"
    bundle = {
        "best_model": model,
        "feature_names": FEATURES,
        "dataset": "Seattle historical listings",
        "model_version": MODEL_VERSION,
    }
    joblib.dump(bundle, model_path)
    artifact_sha256 = _sha256(model_path)

    nested = evaluation["nested_cv"]
    legacy = evaluation["legacy_test"]
    metadata = {
        "model_version": MODEL_VERSION,
        "metadata_schema_version": "1.0",
        "model_type": "HistGradientBoostingRegressor",
        "deployed_model": "hist_gradient_boosting_log_target",
        "deployed_model_type": "HistGradientBoosting with log-target transformation",
        "status": "experimental",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "target_transform": {"forward": "log1p", "inverse": "expm1"},
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "engineered_features": [
            "has_lot_data", "lot_to_house_ratio", "beds_plus_baths", "sqft_per_bedroom",
            "bathroom_bedroom_ratio", "is_small_unit", "size_category",
        ],
        "training_rows": len(train),
        "test_rows": len(test),
        "dataset_rows": len(train),
        "dataset_version": {
            "train_path": TRAIN_PATH.relative_to(ROOT).as_posix(),
            "train_sha256": _sha256(TRAIN_PATH),
            "legacy_test_path": TEST_PATH.relative_to(ROOT).as_posix(),
            "legacy_test_sha256": _sha256(TEST_PATH),
        },
        "metrics": {
            "mae": legacy["mae"],
            "rmse": legacy["rmse"],
            "r2": legacy["r2"],
            "evaluation_label": "legacy_holdout",
        },
        "nested_cv": nested,
        "prediction_interval": evaluation["prediction_interval"],
        "baseline_comparison": {
            "dummy_cv_mae": 384375.50,
            "dummy_cv_rmse": 876816.24,
            "ridge_cv_mae": 243887.93,
            "ridge_cv_rmse": 693738.68,
            "mae_improvement_over_ridge_percent": evaluation["improvement_over_ridge"]["mae_percent"],
            "rmse_improvement_over_ridge_percent": evaluation["improvement_over_ridge"]["rmse_percent"],
        },
        "hyperparameters": tuning["best_params"],
        "selection_policy": "Selected using bounded nested CV; legacy holdout not used for selection",
        "target_summary": {
            "minimum": float(train[TARGET].min()),
            "median": float(train[TARGET].median()),
            "maximum": float(train[TARGET].max()),
        },
        "typical_feature_ranges": _typical_ranges(train[FEATURES]),
        "explanation_reference": {
            "beds": float(train["beds"].median()),
            "baths": float(train["baths"].median()),
            "size": float(train["size"].median()),
            "lot_size": float(train["lot_size"].median()),
            "zip_code": int(train["zip_code"].mode().iloc[0]),
        },
        "feature_importance": [],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "django": django.get_version(),
        },
        "artifact": {
            "format": "joblib",
            "sha256": artifact_sha256,
            "relative_path": model_path.relative_to(ROOT).as_posix(),
        },
        "experiment_tracking": {"provider": "mlflow", "enabled": False},
        "git": _git_state(),
        "limitations": [
            "Only five source fields are available.",
            "Luxury homes above $2M are sparse and commonly underestimated.",
            "The legacy test set was examined by older project phases.",
            "This is an experimental estimate, not a formal appraisal.",
        ],
    }
    if track_mlflow:
        log_mlflow_run(
            metadata,
            model_path,
            artifacts=(TUNING_PATH, EVALUATION_PATH, ROOT / "reports" / "MODEL_EVALUATION.md"),
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )
        metadata["experiment_tracking"]["enabled"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata_sha256 = _sha256(metadata_path)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if REGISTRY_PATH.exists() else {"models": {}}
    for entry in registry["models"].values():
        if entry.get("status") == "active":
            entry["status"] = "rollback"
    registry["models"][MODEL_VERSION] = {
        "artifact": f"versions/{MODEL_VERSION}/model.joblib",
        "metadata": f"versions/{MODEL_VERSION}/metadata.json",
        "status": "active",
        "model_type": metadata["model_type"],
        "created_at": metadata["training_date"],
        "artifact_sha256": artifact_sha256,
        "metadata_sha256": metadata_sha256,
    }
    registry["active_version"] = MODEL_VERSION
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    shutil.copyfile(model_path, COMPATIBILITY_MODEL_PATH)
    shutil.copyfile(metadata_path, COMPATIBILITY_METADATA_PATH)
    print(f"Registered Seattle model {MODEL_VERSION} at {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-mlflow", action="store_true", help="Log this run to optional MLflow")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"),
                        help="Override MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "house-price-seattle"))
    args = parser.parse_args()
    main(args.track_mlflow, args.tracking_uri, args.experiment_name)
