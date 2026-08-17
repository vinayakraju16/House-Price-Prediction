"""Leakage-safe benchmark for the official King County Seattle cohort."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline

from ml.data.king_county import MODEL_FEATURES
from ml.features.build_features import build_base_preprocessor, build_king_county_preprocessor

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "king-county" / "sales.csv"
RESULT_PATH = ROOT / "reports" / "king_county_benchmark.json"
TABLE_PATH = ROOT / "reports" / "king_county_benchmark.csv"
TARGET = "price"
BASE_FEATURES = ["beds", "baths", "size", "lot_size", "zip_code"]
SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: object
    rich: bool
    log_target: bool


def candidates() -> list[Candidate]:
    common = {"max_iter": 250, "max_leaf_nodes": 31, "learning_rate": 0.06,
              "l2_regularization": 10, "min_samples_leaf": 30, "random_state": 42}
    return [
        Candidate("Dummy median", DummyRegressor(strategy="median"), False, False),
        Candidate("Ridge rich", Ridge(alpha=10.0), True, True),
        Candidate("HistGradientBoosting five-field", HistGradientBoostingRegressor(**common), False, True),
        Candidate("HistGradientBoosting rich raw", HistGradientBoostingRegressor(**common), True, False),
        Candidate("HistGradientBoosting rich log", HistGradientBoostingRegressor(**common), True, True),
        Candidate(
            "RandomForest rich log",
            RandomForestRegressor(n_estimators=200, max_depth=18, min_samples_leaf=3,
                                  max_features=0.8, n_jobs=-1, random_state=42),
            True,
            True,
        ),
    ]


def build_model(candidate: Candidate):
    preprocessor = build_king_county_preprocessor() if candidate.rich else build_base_preprocessor()
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", candidate.estimator)])
    if candidate.log_target:
        return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
    return pipeline


def temporal_group_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve 2025+ sales and remove their parcels from development data."""
    holdout = data.loc[data["sale_year"] >= 2025].copy()
    holdout_parcels = set(holdout["parcel_id"])
    development = data.loc[
        (data["sale_year"] <= 2024) & ~data["parcel_id"].isin(holdout_parcels)
    ].copy()
    if development.empty or holdout.empty:
        raise ValueError("The dataset must contain both pre-2025 development and 2025+ holdout rows")
    return development, holdout


def evaluate(candidate: Candidate, development: pd.DataFrame, holdout: pd.DataFrame) -> tuple[dict, object]:
    feature_names = MODEL_FEATURES if candidate.rich else BASE_FEATURES
    model = build_model(candidate)
    cv = GroupKFold(n_splits=5)
    started = time.perf_counter()
    scores = cross_validate(
        model,
        development[feature_names],
        development[TARGET],
        groups=development["parcel_id"],
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )
    model.fit(development[feature_names], development[TARGET])
    prediction = model.predict(holdout[feature_names])
    result = {
        "model": candidate.name,
        "features": "rich" if candidate.rich else "five-field",
        "target_transform": "log1p" if candidate.log_target else "raw",
        "cv_mae_mean": float(-scores["test_mae"].mean()),
        "cv_mae_std": float(scores["test_mae"].std()),
        "cv_rmse_mean": float(-scores["test_rmse"].mean()),
        "cv_rmse_std": float(scores["test_rmse"].std()),
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std()),
        "test_mae": float(mean_absolute_error(holdout[TARGET], prediction)),
        "test_rmse": float(mean_squared_error(holdout[TARGET], prediction) ** 0.5),
        "test_r2": float(r2_score(holdout[TARGET], prediction)),
        "seconds": round(time.perf_counter() - started, 2),
    }
    return result, model


def main() -> None:
    data = pd.read_csv(DATA_PATH, dtype={"zip_code": "string", "heat_system": "string",
                                        "property_type": "string"})
    development, holdout = temporal_group_split(data)
    results = []
    for candidate in candidates():
        print(f"Evaluating {candidate.name}", flush=True)
        result, _ = evaluate(candidate, development, holdout)
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)
    best = min(results, key=lambda item: item["cv_rmse_mean"])
    payload = {
        "selection_metric": "mean five-fold GroupKFold RMSE",
        "split": "Development through 2024; 2025-2026 temporal holdout; holdout parcels excluded from development",
        "development_rows": len(development),
        "development_parcels": int(development["parcel_id"].nunique()),
        "holdout_rows": len(holdout),
        "holdout_parcels": int(holdout["parcel_id"].nunique()),
        "results": results,
        "best": best,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(TABLE_PATH, index=False)
    print(f"Wrote {RESULT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
