"""Evaluate, document, fit, and register the King County model as version 3.0.0."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.data.king_county import ARCHIVES, MODEL_FEATURES
from ml.training.benchmark_king_county import temporal_group_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "king-county" / "sales.csv"
AUDIT_PATH = ROOT / "reports" / "king_county_data_audit.json"
BENCHMARK_PATH = ROOT / "reports" / "king_county_benchmark.json"
TUNING_PATH = ROOT / "reports" / "king_county_tuning.json"
TUNED_MODEL_PATH = ROOT / "reports" / "king_county_tuned_model.joblib"
EVALUATION_PATH = ROOT / "reports" / "king_county_model_evaluation.json"
REPORT_PATH = ROOT / "reports" / "MODEL_EVALUATION.md"
MODEL_DIR = ROOT / "backend" / "mlmodels"
VERSION = "3.0.0"
REQUIRED_FEATURES = ["beds", "baths", "size", "lot_size", "zip_code"]
SERVER_FEATURES = ["sale_year", "sale_month"]
OPTIONAL_FEATURES = [name for name in MODEL_FEATURES if name not in REQUIRED_FEATURES + SERVER_FEATURES]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(actual, prediction) -> dict:
    return {
        "rows": len(actual),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
    }


def _segments(frame: pd.DataFrame) -> dict:
    result = {}
    for name, group in frame.groupby("segment", observed=True):
        result[str(name)] = _metrics(group["price"], group["prediction"])
    return result


def _default_values(data: pd.DataFrame) -> dict:
    defaults = {}
    categorical = {"zip_code", "heat_system", "property_type"}
    for feature in MODEL_FEATURES:
        if feature == "year_renovated":
            defaults[feature] = None
        elif feature in categorical:
            defaults[feature] = str(data[feature].mode(dropna=True).iloc[0])
        else:
            defaults[feature] = float(data[feature].median())
    now = datetime.now(timezone.utc)
    defaults.update({"sale_year": now.year, "sale_month": now.month})
    return defaults


def _typical_ranges(data: pd.DataFrame) -> dict:
    ranges = {}
    for feature in MODEL_FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce").dropna()
        if values.empty:
            continue
        ranges[feature] = {
            "minimum": float(values.min()),
            "p05": float(values.quantile(0.05)),
            "p95": float(values.quantile(0.95)),
            "maximum": float(values.max()),
        }
    return ranges


def _render_report(evaluation: dict, benchmark: dict, tuning: dict) -> str:
    rows = "\n".join(
        f"| {row['model']} | ${row['cv_mae_mean']:,.0f} | ${row['cv_rmse_mean']:,.0f} | "
        f"{row['cv_r2_mean']:.3f} | ${row['test_mae']:,.0f} | ${row['test_rmse']:,.0f} | {row['test_r2']:.3f} |"
        for row in benchmark["results"]
    )
    segments = "\n".join(
        f"| {name} | {values['rows']:,} | ${values['mae']:,.0f} | ${values['rmse']:,.0f} | {values['r2']:.3f} |"
        for name, values in evaluation["price_segments"].items()
    )
    return f"""# King County Model Evaluation

The model was selected using five-fold `GroupKFold` cross-validation, grouping by parcel. Sales from 2025–2026 form a temporal holdout, and parcels appearing in that holdout were excluded from development data.

## Model comparison

| Model | CV MAE | CV RMSE | CV R² | Holdout MAE | Holdout RMSE | Holdout R² |
|---|---:|---:|---:|---:|---:|---:|
{rows}

The bounded 12-trial randomized search selected a rich log-target HistGradientBoosting model with CV RMSE **${tuning['best_cv_rmse']:,.0f}**. Its untouched temporal-holdout metrics were MAE **${evaluation['overall']['mae']:,.0f}**, RMSE **${evaluation['overall']['rmse']:,.0f}**, and R² **{evaluation['overall']['r2']:.3f}**.

## Error by sale-price band

| Segment | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
{segments}

## Uncertainty

The API interval uses the 5th and 95th percentiles of temporal-holdout residuals. Its measured holdout coverage is **{tuning['prediction_interval']['empirical_coverage']:.1%}** for a nominal 90% interval. This is an empirical population-level range, not a formal appraisal guarantee.

## Limitations

- Seattle coverage is approximated using King County ZIP codes 98101–98199.
- Public assessor sales and building attributes can lag corrections and renovations.
- Luxury homes remain harder to estimate and dominate RMSE.
- Current implementation has no precise latitude/longitude or neighborhood field.
"""


def main() -> None:
    data = pd.read_csv(DATA_PATH, dtype={"zip_code": "string", "heat_system": "string",
                                        "property_type": "string"})
    development, holdout = temporal_group_split(data)
    model = joblib.load(TUNED_MODEL_PATH)
    prediction = model.predict(holdout[MODEL_FEATURES])
    evaluation_frame = holdout[["parcel_id", "sale_date", "price", "zip_code", "beds"]].copy()
    evaluation_frame["prediction"] = prediction
    evaluation_frame["residual"] = evaluation_frame["price"] - prediction
    evaluation_frame["absolute_error"] = evaluation_frame["residual"].abs()
    evaluation_frame["segment"] = pd.cut(
        evaluation_frame["price"],
        bins=[0, 500_000, 1_000_000, 2_000_000, np.inf],
        labels=["Below $500K", "$500K-$1M", "$1M-$2M", "$2M+"],
    )

    importance_sample = holdout.sample(n=min(3000, len(holdout)), random_state=42)
    importance = permutation_importance(
        model,
        importance_sample[MODEL_FEATURES],
        importance_sample["price"],
        scoring="neg_root_mean_squared_error",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )
    feature_importance = sorted(
        [
            {"feature": feature, "importance": float(score), "std": float(std)}
            for feature, score, std in zip(
                MODEL_FEATURES, importance.importances_mean, importance.importances_std, strict=True
            )
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )
    evaluation = {
        "overall": _metrics(holdout["price"], prediction),
        "price_segments": _segments(evaluation_frame),
        "zip_segments": {
            str(name): _metrics(group["price"], group["prediction"])
            for name, group in evaluation_frame.groupby("zip_code") if len(group) >= 30
        },
        "bedroom_segments": {
            str(name): _metrics(group["price"], group["prediction"])
            for name, group in evaluation_frame.groupby("beds") if len(group) >= 30
        },
        "largest_errors": evaluation_frame.nlargest(20, "absolute_error")[
            ["sale_date", "price", "prediction", "residual", "zip_code", "beds"]
        ].to_dict(orient="records"),
        "feature_importance": feature_importance,
    }
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8", newline="\n")
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    tuning = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    REPORT_PATH.write_text(_render_report(evaluation, benchmark, tuning), encoding="utf-8", newline="\n")

    model.fit(data[MODEL_FEATURES], data["price"])
    recent_comparables = (
        data.loc[data["sale_year"] >= 2023, [*MODEL_FEATURES, "price", "sale_date"]]
        .tail(20_000)
        .to_dict(orient="records")
    )
    version_dir = MODEL_DIR / "versions" / VERSION
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.joblib"
    metadata_path = version_dir / "metadata.json"
    bundle = {
        "best_model": model,
        "feature_names": MODEL_FEATURES,
        "required_features": REQUIRED_FEATURES,
        "optional_features": OPTIONAL_FEATURES,
        "comparables": recent_comparables,
        "model_version": VERSION,
    }
    joblib.dump(bundle, model_path, compress=3)
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_version": VERSION,
        "metadata_schema_version": "2.0",
        "status": "experimental",
        "model_type": "HistGradientBoostingRegressor",
        "deployed_model": "king_county_hist_gradient_boosting_log_target",
        "deployed_model_type": "HistGradientBoosting with log-target transformation",
        "training_date": trained_at,
        "dataset": "King County Assessor real-property sales, residential building, and parcel extracts",
        "dataset_rows": len(data),
        "training_rows": len(data),
        "test_rows": len(holdout),
        "features": MODEL_FEATURES,
        "required_features": REQUIRED_FEATURES,
        "optional_features": OPTIONAL_FEATURES,
        "server_features": SERVER_FEATURES,
        "feature_count": len(MODEL_FEATURES),
        "feature_defaults": _default_values(data),
        "typical_feature_ranges": _typical_ranges(data),
        "explanation_reference": _default_values(data),
        "feature_importance": feature_importance,
        "metrics": {**evaluation["overall"], "evaluation_label": "2025-2026 temporal holdout"},
        "cross_validation": {
            "type": "GroupKFold", "folds": 5, "group": "parcel_id",
            "rmse": tuning["best_cv_rmse"],
        },
        "prediction_interval": tuning["prediction_interval"],
        "hyperparameters": tuning["best_params"],
        "target_summary": {
            "minimum": float(data["price"].min()), "median": float(data["price"].median()),
            "maximum": float(data["price"].max()),
        },
        "dataset_version": {
            "processed_path": DATA_PATH.relative_to(ROOT).as_posix(),
            "processed_sha256": _sha256(DATA_PATH),
            "source_archives": {
                name: {"path": f"data/raw/king-county/{archive}",
                       "sha256": _sha256(ROOT / "data" / "raw" / "king-county" / archive)}
                for name, (archive, _) in ARCHIVES.items()
            },
        },
        "privacy": "Owner names, street addresses, and recording identifiers are excluded at read time.",
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "artifact": {"format": "joblib", "relative_path": model_path.relative_to(ROOT).as_posix()},
        "limitations": [
            "Seattle coverage is approximated using ZIP codes 98101-98199.",
            "The estimate is experimental and is not a formal appraisal.",
            "Luxury properties have larger absolute errors.",
        ],
    }
    model_sha = _sha256(model_path)
    metadata["artifact"]["sha256"] = model_sha
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    metadata_sha = _sha256(metadata_path)

    registry_path = MODEL_DIR / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for entry in registry["models"].values():
        if entry.get("status") == "active":
            entry["status"] = "rollback"
    registry["models"][VERSION] = {
        "artifact": f"versions/{VERSION}/model.joblib",
        "metadata": f"versions/{VERSION}/metadata.json",
        "status": "active",
        "model_type": metadata["model_type"],
        "created_at": trained_at,
        "artifact_sha256": model_sha,
        "metadata_sha256": metadata_sha,
    }
    registry["active_version"] = VERSION
    registry["updated_at"] = trained_at
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8", newline="\n")
    shutil.copyfile(model_path, MODEL_DIR / "trained_models.pkl")
    shutil.copyfile(metadata_path, MODEL_DIR / "metadata.json")
    print(f"Registered King County model {VERSION}; holdout R2={evaluation['overall']['r2']:.3f}")


if __name__ == "__main__":
    main()
