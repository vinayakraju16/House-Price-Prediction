"""Export the registered King County model so the React app can run it in the browser.

The deployed frontend is a static Cloudflare Workers site with no backend. This command
reads the fitted parameters of the registered scikit-learn pipeline (median imputation,
standard scaling, one-hot categories, and every HistGradientBoosting tree) and writes them
as JSON that ``frontend/src/model/predictor.js`` evaluates with the same arithmetic.

    python -m ml.training.export_browser_model            # model + comparables
    python -m ml.training.export_browser_model --fixtures # also write parity fixtures

``--fixtures`` scores random properties with the real scikit-learn pipeline and the
backend's own explanation and comparable-search functions, and stores the results so the
frontend test suite can prove the JavaScript port returns the same numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
MODEL_DIR = BACKEND / "mlmodels"
OUTPUT_DIR = ROOT / "frontend" / "src" / "model"
FIXTURE_DIR = OUTPUT_DIR / "__fixtures__"
FORMAT = "havenvalue-browser-model"
FORMAT_VERSION = 1
SHAPLEY_PERMUTATIONS = 128
SHAPLEY_SEED = 42
COMPARABLE_COLUMNS = (
    "beds", "baths", "size", "lot_size", "zip_code", "price", "sale_date", "year_built",
)
MODEL_INFO_FIELDS = (
    "model_version", "status", "deployed_model", "deployed_model_type", "training_date",
    "dataset_rows", "training_rows", "test_rows", "feature_count", "metrics",
    "cross_validation", "prediction_interval", "target_summary",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_loss_unpickle_shims() -> None:
    """Let newer scikit-learn releases unpickle the training-only loss object.

    The artifact was saved with scikit-learn 1.4.2. Later releases renamed the Cython
    pickle helpers for loss objects; prediction never uses the loss, so a no-op is safe.
    With the pinned 1.4.2 runtime the helpers exist and nothing is replaced.
    """
    from sklearn._loss import _loss

    for name in ("CyHalfSquaredError", "CyAbsoluteError", "CyHalfPoissonLoss"):
        helper = f"__pyx_unpickle_{name}"
        if not hasattr(_loss, helper):
            setattr(_loss, helper, lambda *args, **kwargs: None)


def _patch_predict_attributes(estimator) -> None:
    """Add attributes newer scikit-learn expects at predict time; no-op on 1.4.2."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if isinstance(estimator, Pipeline):
        if not hasattr(estimator, "transform_input"):
            estimator.transform_input = None
        for _, step in estimator.steps:
            _patch_predict_attributes(step)
    if isinstance(estimator, SimpleImputer) and not hasattr(estimator, "_fill_dtype"):
        estimator._fill_dtype = estimator._fit_dtype
    if hasattr(estimator, "regressor_"):
        _patch_predict_attributes(estimator.regressor_)
    for item in getattr(estimator, "transformers_", ()):
        if not isinstance(item[1], str):
            _patch_predict_attributes(item[1])


def load_registered(version: str | None = None):
    registry = json.loads((MODEL_DIR / "model_registry.json").read_text(encoding="utf-8"))
    version = version or registry["active_version"]
    entry = registry["models"][version]
    artifact_path = MODEL_DIR / entry["artifact"]
    metadata_path = MODEL_DIR / entry["metadata"]
    if entry.get("artifact_sha256") and _sha256(artifact_path) != entry["artifact_sha256"]:
        raise RuntimeError(f"{artifact_path} does not match the registered checksum.")

    sys.path.insert(0, str(BACKEND))
    _install_loss_unpickle_shims()
    import joblib

    bundle = joblib.load(artifact_path)
    _patch_predict_attributes(bundle["best_model"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return version, entry, bundle, metadata


def _finite(value):
    """JSON cannot hold NaN; null means missing and the JS side maps it back to NaN."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _tree_payload(regressor) -> dict:
    trees = []
    for iteration in regressor._predictors:
        if len(iteration) != 1:
            raise RuntimeError("Only single-output regression trees are supported.")
        nodes = iteration[0].nodes
        if nodes["is_categorical"].any():
            raise RuntimeError(
                "Native categorical splits are not supported by the browser runtime."
            )
        leaf = nodes["is_leaf"].astype(bool)
        trees.append({
            # Leaves keep feature -1 so the evaluator can stop without a separate flag array.
            "feature": np.where(leaf, -1, nodes["feature_idx"]).astype(int).tolist(),
            "threshold": nodes["num_threshold"].astype(float).tolist(),
            "missing_left": nodes["missing_go_to_left"].astype(int).tolist(),
            "left": nodes["left"].astype(int).tolist(),
            "right": nodes["right"].astype(int).tolist(),
            "value": nodes["value"].astype(float).tolist(),
        })
    baseline = np.asarray(regressor._baseline_prediction, dtype=float).ravel()
    return {"baseline": float(baseline[0]), "trees": trees}


def build_browser_model(version, entry, bundle, metadata) -> dict:
    model = bundle["best_model"]
    if model.func.__name__ != "log1p" or model.inverse_func.__name__ != "expm1":
        raise RuntimeError("The browser runtime expects a log1p target transformation.")
    preprocessor, regressor = (step for _, step in model.regressor_.steps)
    engineer, columns = (step for _, step in preprocessor.steps)
    if type(engineer).__name__ != "KingCountyFeatureEngineer":
        raise RuntimeError("The browser runtime implements KingCountyFeatureEngineer only.")

    transformers = {name: (pipeline, list(cols)) for name, pipeline, cols in columns.transformers_}
    numeric_pipeline, numeric_columns = transformers["numeric"]
    categorical_pipeline, categorical_columns = transformers["categorical"]
    numeric_imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    encoder = categorical_pipeline.named_steps["onehot"]
    if encoder.handle_unknown != "ignore" or encoder.drop is not None:
        raise RuntimeError("The browser runtime expects OneHotEncoder(handle_unknown='ignore').")

    rng = np.random.default_rng(SHAPLEY_SEED)
    feature_count = len(metadata["features"])
    permutations = [rng.permutation(feature_count).tolist() for _ in range(SHAPLEY_PERMUTATIONS)]

    model_info = {key: metadata.get(key) for key in MODEL_INFO_FIELDS}
    model_info["feature_importance"] = metadata.get("feature_importance", [])[:6]
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model_version": version,
        "artifact_sha256": entry.get("artifact_sha256"),
        "features": metadata["features"],
        "feature_defaults": {
            key: _finite(value) for key, value in metadata["feature_defaults"].items()
        },
        "typical_feature_ranges": metadata["typical_feature_ranges"],
        "explanation_reference": {
            key: _finite(value) for key, value in metadata["explanation_reference"].items()
        },
        "prediction_interval": metadata["prediction_interval"],
        "model_info": model_info,
        "preprocessing": {
            "numeric": {
                "columns": numeric_columns,
                "impute": numeric_imputer.statistics_.astype(float).tolist(),
                "mean": scaler.mean_.astype(float).tolist(),
                "scale": scaler.scale_.astype(float).tolist(),
            },
            "categorical": {
                "columns": categorical_columns,
                "impute": [str(value) for value in categorical_imputer.statistics_],
                "categories": [[str(value) for value in values] for values in encoder.categories_],
            },
        },
        "target_transform": "log1p",
        "regressor": _tree_payload(regressor),
        "shapley": {"seed": SHAPLEY_SEED, "permutations": permutations},
    }


def build_comparables(bundle, metadata) -> dict:
    data = pd.DataFrame(bundle["comparables"])
    columns = {}
    for column in COMPARABLE_COLUMNS:
        series = data[column]
        if column == "zip_code":
            columns[column] = series.astype(int).tolist()
        elif column == "sale_date":
            columns[column] = series.astype(str).tolist()
        elif column in ("beds", "year_built"):
            columns[column] = series.astype(int).tolist()
        else:
            columns[column] = [_finite(value) for value in series.astype(float)]
    # find_comparable_properties scales log sizes by quantiles of the full dataset; they
    # never change between requests, so they are computed once here with numpy's method.
    log_scales = {}
    for feature in ("size", "lot_size"):
        values = np.log1p(data[feature].to_numpy(dtype=float))
        log_scales[feature] = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    return {"rows": len(data), "columns": columns, "log_scales": log_scales}


def _random_payloads(rng, count):
    zips = ["98102", "98103", "98105", "98115", "98117", "98118", "98122", "98125",
            "98133", "98144", "98155", "98168", "98177", "98199", "98101", "98104"]
    payloads = []
    for _ in range(count):
        payload = {
            "beds": int(rng.integers(0, 8)),
            "baths": float(rng.choice([0.5, 1, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4.5])),
            "size": int(rng.integers(400, 7000)),
            "lot_size": int(rng.choice([0, 800, 2500, 4000, 6000, 9000, 20000, 120000])),
            "zip_code": int(rng.choice(zips)),
        }
        optional = {
            "stories": float(rng.choice([1, 1.5, 2, 3])),
            "grade": int(rng.integers(4, 14)),
            "condition": int(rng.integers(1, 6)),
            "year_built": int(rng.integers(1900, 2026)),
            "finished_basement_sqft": int(rng.integers(0, 1500)),
            "garage_sqft": int(rng.integers(0, 800)),
            "fireplaces": int(rng.integers(0, 4)),
            "has_view": int(rng.integers(0, 2)),
            "heat_system": int(rng.integers(1, 10)),
            "property_type": int(rng.choice([2, 3, 29, 57, 999])),
        }
        for name, value in optional.items():
            if rng.random() < 0.6:
                payload[name] = value
        if "year_built" in payload and rng.random() < 0.4:
            payload["year_renovated"] = int(rng.integers(payload["year_built"], 2027))
        payloads.append(payload)
    return payloads


def build_parity_fixtures(bundle, metadata, comparables_frame, count=60, year=2026, month=10):
    """Score payloads with scikit-learn and the backend's own estimate helpers."""
    from django.conf import settings

    if not settings.configured:
        settings.configure(BASE_DIR=BACKEND, MODEL_VERSION="")
    from api import services

    class FixedDateTime:
        @staticmethod
        def now(tz=None):
            from datetime import datetime

            return datetime(year, month, 15, tzinfo=tz)

    services.datetime = FixedDateTime
    services.model_metadata = lambda version=None: metadata
    services.comparable_data = lambda: comparables_frame
    model = bundle["best_model"]
    rng = np.random.default_rng(2026)
    cases = []
    for payload in _random_payloads(rng, count):
        frame = services.build_input_frame(payload)
        prediction = float(model.predict(frame)[0])
        explanation = services.explain_prediction(model, frame, metadata)
        comparables = services.find_comparable_properties(frame, metadata)
        cases.append({
            "payload": payload,
            "prediction": prediction,
            # services.explain_prediction reads the row back with iloc[0], which upcasts an
            # all-numeric row to floats; categorical codes then become "98102.0" and miss the
            # one-hot vocabulary. A row is all-numeric only when heat_system, property_type
            # and year_renovated are all sent (otherwise a string default or a null keeps an
            # object column), so the fixture records whether the Python paths ended at the
            # real prediction, and the frontend test compares factors only when they did.
            "explanation_consistent": bool(
                abs(explanation["prediction"] - prediction) <= 1e-6 * max(prediction, 1.0)
            ),
            "base_value": explanation["base_value"],
            "contributions": {
                item["feature"]: item["contribution"] for item in explanation["top_factors"]
            },
            "comparable_prices": [item["price"] for item in comparables],
            "comparable_similarity": [item["similarity"] for item in comparables],
        })
    return {"sale_year": year, "sale_month": month, "cases": cases}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KiB)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="registry version to export (default: active)")
    parser.add_argument("--fixtures", action="store_true", help="also write parity fixtures")
    args = parser.parse_args(argv)

    version, entry, bundle, metadata = load_registered(args.version)
    browser_model = build_browser_model(version, entry, bundle, metadata)
    comparables = build_comparables(bundle, metadata)
    _write_json(OUTPUT_DIR / f"king-county-{version}.json", browser_model)
    _write_json(OUTPUT_DIR / f"comparables-{version}.json", comparables)
    if args.fixtures:
        fixtures = build_parity_fixtures(bundle, metadata, pd.DataFrame(bundle["comparables"]))
        _write_json(FIXTURE_DIR / f"parity-{version}.json", fixtures)


if __name__ == "__main__":
    main()
