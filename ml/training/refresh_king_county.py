"""Refresh the Seattle (King County) model with newly published assessor sales.

    python -m ml.data.king_county                         # re-prepare from new raw extracts
    python -m ml.training.refresh_king_county             # evaluate + export browser model
    python -m ml.training.refresh_king_county --register  # also save joblib + registry entry

Steps:
1. Score the currently deployed model on sales it has never seen (after ``--since``,
   the last sale date in its training data). This is the honest "how is it doing now".
2. Evaluate the same tuned recipe on a temporal holdout (``HOLDOUT_START`` onward, holdout
   parcels removed from development) and derive the residual-quantile range from it.
3. Refit on every sale and export the browser model, comparables and parity fixtures.

``--register`` also writes ``backend/mlmodels/versions/<version>/`` and activates it in the
registry. Run it with the pinned runtime (scikit-learn 1.4.2) so the backend can load it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from ml.data.king_county import ARCHIVES, MODEL_FEATURES
from ml.features.build_features import build_king_county_preprocessor
from ml.training import export_browser_model as exporter
from ml.training.train_king_county import (
    OPTIONAL_FEATURES,
    REQUIRED_FEATURES,
    SERVER_FEATURES,
    _default_values,
    _sha256,
    _typical_ranges,
)
from ml.training.train_wake_county import metrics

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "king-county" / "sales.csv"
TUNING_PATH = ROOT / "reports" / "king_county_tuning.json"
RESULT_PATH = ROOT / "reports" / "king_county_refresh.json"
MODEL_DIR = ROOT / "backend" / "mlmodels"
VERSION = "3.1.0"
PREVIOUS_VERSION = "3.0.0"
HOLDOUT_START = "2026-01-01"
DEFAULT_SINCE = "2026-07-22"


def tuned_params() -> dict:
    params = json.loads(TUNING_PATH.read_text(encoding="utf-8"))["best_params"]
    return {key.rsplit("__", 1)[-1]: value for key, value in params.items()} | {"random_state": 42}


def build_model():
    pipeline = Pipeline([("preprocessor", build_king_county_preprocessor()),
                         ("regressor", HistGradientBoostingRegressor(**tuned_params()))])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, dtype={"parcel_id": str, "zip_code": "string",
                                        "heat_system": "string", "property_type": "string"})


def score_previous(data: pd.DataFrame, since: str) -> dict | None:
    unseen = data.loc[data["sale_date"] > since]
    if unseen.empty:
        return None
    _, _, bundle, _ = exporter.load_registered(PREVIOUS_VERSION)
    prediction = bundle["best_model"].predict(unseen[MODEL_FEATURES])
    return {"since": since, **metrics(unseen["price"], prediction),
            "date_range": [unseen["sale_date"].min(), unseen["sale_date"].max()]}


def evaluate(data: pd.DataFrame) -> tuple[dict, dict]:
    holdout = data.loc[data["sale_date"] >= HOLDOUT_START]
    development = data.loc[(data["sale_date"] < HOLDOUT_START)
                           & ~data["parcel_id"].isin(set(holdout["parcel_id"]))]
    model = build_model().fit(development[MODEL_FEATURES], development["price"])
    prediction = model.predict(holdout[MODEL_FEATURES])
    residuals = holdout["price"].to_numpy() - prediction
    low, high = np.quantile(residuals, [0.05, 0.95])
    interval = {
        "method": f"{HOLDOUT_START[:4]} temporal-holdout residual quantiles",
        "nominal_coverage": 0.9,
        "lower_residual_quantile": float(low),
        "upper_residual_quantile": float(high),
        "empirical_coverage": float(np.mean((residuals >= low) & (residuals <= high))),
    }
    evaluation = {"holdout_start": HOLDOUT_START, "development_rows": len(development),
                  **metrics(holdout["price"], prediction)}
    return evaluation, interval


def build_bundle_and_metadata(data, evaluation, interval, previous_unseen):
    model = build_model().fit(data[MODEL_FEATURES], data["price"])
    comparables = (data.loc[data["sale_year"] >= 2023, [*MODEL_FEATURES, "price", "sale_date"]]
                   .tail(20_000).to_dict(orient="records"))
    bundle = {"best_model": model, "feature_names": MODEL_FEATURES,
              "required_features": REQUIRED_FEATURES, "optional_features": OPTIONAL_FEATURES,
              "comparables": comparables, "model_version": VERSION}
    previous = json.loads((MODEL_DIR / "versions" / PREVIOUS_VERSION / "metadata.json")
                          .read_text(encoding="utf-8"))
    defaults = _default_values(data)
    metadata = {
        **{key: previous[key] for key in ("metadata_schema_version", "model_type",
                                         "deployed_model", "deployed_model_type")},
        "model_version": VERSION,
        "status": "experimental",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset": previous["dataset"],
        "dataset_rows": len(data), "training_rows": len(data),
        "test_rows": evaluation["rows"],
        "features": MODEL_FEATURES, "required_features": REQUIRED_FEATURES,
        "optional_features": OPTIONAL_FEATURES, "server_features": SERVER_FEATURES,
        "feature_count": len(MODEL_FEATURES),
        "feature_defaults": defaults,
        "typical_feature_ranges": _typical_ranges(data),
        "explanation_reference": defaults,
        "feature_importance": previous.get("feature_importance", []),
        "metrics": {key: evaluation[key] for key in ("rows", "mae", "rmse", "r2")}
        | {"median_ape": evaluation["median_ape"],
           "evaluation_label": f"{HOLDOUT_START[:4]} temporal holdout"},
        "previous_model_on_new_sales": previous_unseen,
        "cross_validation": previous.get("cross_validation"),
        "prediction_interval": interval,
        "hyperparameters": tuned_params(),
        "target_summary": {"minimum": float(data["price"].min()),
                           "median": float(data["price"].median()),
                           "maximum": float(data["price"].max())},
        "dataset_version": {
            "processed_path": DATA_PATH.relative_to(ROOT).as_posix(),
            "processed_sha256": _sha256(DATA_PATH),
            "date_range": [data["sale_date"].min(), data["sale_date"].max()],
            "source_archives": {name: f"data/raw/king-county/{archive}"
                                for name, (archive, _) in ARCHIVES.items()},
        },
        "privacy": previous["privacy"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                    "joblib": joblib.__version__},
        "limitations": previous["limitations"],
    }
    return bundle, metadata


def register(bundle, metadata) -> dict:
    version_dir = MODEL_DIR / "versions" / VERSION
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    metadata["artifact"] = {"format": "joblib",
                            "relative_path": model_path.relative_to(ROOT).as_posix(),
                            "sha256": _sha256(model_path)}
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    registry_path = MODEL_DIR / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for entry in registry["models"].values():
        if entry.get("status") == "active":
            entry["status"] = "rollback"
    entry = {"artifact": f"versions/{VERSION}/model.joblib",
             "metadata": f"versions/{VERSION}/metadata.json", "status": "active",
             "model_type": metadata["model_type"], "created_at": metadata["training_date"],
             "artifact_sha256": metadata["artifact"]["sha256"],
             "metadata_sha256": _sha256(metadata_path)}
    registry["models"][VERSION] = entry
    registry["active_version"] = VERSION
    registry["updated_at"] = metadata["training_date"]
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8", newline="\n")
    return entry


def _django_stub() -> None:
    """The parity fixtures reuse backend/api/services.py, which imports Django settings and
    cache. When Django is not installed, provide the two objects it touches."""
    try:
        import django  # noqa: F401
        return
    except ImportError:
        pass
    conf = types.ModuleType("django.conf")
    conf.settings = types.SimpleNamespace(configured=True, BASE_DIR=ROOT / "backend",
                                          MODEL_VERSION="")
    cache = types.ModuleType("django.core.cache")
    cache.cache = types.SimpleNamespace(get=lambda *a, **k: None, set=lambda *a, **k: None)
    sys.modules.update({"django": types.ModuleType("django"), "django.conf": conf,
                        "django.core": types.ModuleType("django.core"),
                        "django.core.cache": cache})


def export(bundle, metadata, entry) -> None:
    browser = exporter.build_browser_model(VERSION, entry, bundle, metadata)
    exporter._write_json(exporter.OUTPUT_DIR / f"king-county-{VERSION}.json", browser)
    exporter._write_json(exporter.OUTPUT_DIR / f"comparables-{VERSION}.json",
                         exporter.build_comparables(bundle, metadata))
    _django_stub()
    sys.path.insert(0, str(ROOT / "backend"))
    fixtures = exporter.build_parity_fixtures(bundle, metadata, pd.DataFrame(bundle["comparables"]))
    exporter._write_json(exporter.FIXTURE_DIR / f"parity-{VERSION}.json", fixtures)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", default=DEFAULT_SINCE,
                        help="last sale date the previous model was trained on")
    parser.add_argument("--register", action="store_true",
                        help="save joblib + metadata and activate it in the model registry")
    args = parser.parse_args(argv)

    data = load_data()
    previous_unseen = score_previous(data, args.since)
    evaluation, interval = evaluate(data)
    print(json.dumps({"previous_model_on_new_sales": previous_unseen,
                      "new_recipe_holdout": evaluation}, indent=2), flush=True)
    bundle, metadata = build_bundle_and_metadata(data, evaluation, interval, previous_unseen)
    if args.register:
        entry = register(bundle, metadata)
    else:
        path = ROOT / "reports" / f"king_county_{VERSION}_model.joblib"
        joblib.dump(bundle, path, compress=3)
        entry = {"artifact_sha256": _sha256(path)}
        path.unlink()
    export(bundle, metadata, entry)
    RESULT_PATH.write_text(json.dumps({
        "version": VERSION, "previous_version": PREVIOUS_VERSION,
        "previous_model_on_new_sales": previous_unseen, "holdout": evaluation,
        "prediction_interval": interval, "rows": len(data),
        "date_range": metadata["dataset_version"]["date_range"],
        "registered": bool(args.register), "runtime": metadata["runtime"],
    }, indent=2), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
