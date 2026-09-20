"""CV-first Seattle model comparison with a quarantined legacy holdout.

Run from the repository root:
    python -m ml.training.benchmark
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from ml.data.validation import validate_seattle_dataframe
from ml.features.build_features import build_base_preprocessor, build_engineered_preprocessor

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "Seattle" / "train_cleaned.csv"
TEST_PATH = ROOT / "data" / "Seattle" / "test_cleaned.csv"
JSON_PATH = ROOT / "reports" / "model_benchmark_v2.json"
CSV_PATH = ROOT / "reports" / "model_benchmark_v2.csv"
REPORT_PATH = ROOT / "reports" / "MODEL_BENCHMARK.md"
FEATURES = ["beds", "baths", "size", "lot_size", "zip_code"]
TARGET = "price"
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: object
    feature_set: str = "base"
    log_target: bool = False


def candidates() -> list[Candidate]:
    """Return deliberately bounded, dependency-light benchmark candidates."""
    return [
        Candidate("Dummy median", DummyRegressor(strategy="median")),
        Candidate("Linear Regression", LinearRegression()),
        Candidate("Linear Regression", LinearRegression(), log_target=True),
        Candidate("Ridge", Ridge(alpha=10.0)),
        Candidate("Ridge", Ridge(alpha=10.0), log_target=True),
        Candidate("Ridge", Ridge(alpha=10.0), feature_set="engineered"),
        Candidate("Ridge", Ridge(alpha=10.0), feature_set="engineered", log_target=True),
        Candidate("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20_000)),
        Candidate("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20_000), log_target=True),
        Candidate(
            "Random Forest",
            RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=1),
            feature_set="engineered",
        ),
        Candidate(
            "Random Forest",
            RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=1),
            feature_set="engineered",
            log_target=True,
        ),
        Candidate(
            "Gradient Boosting",
            GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42),
            feature_set="engineered",
        ),
        Candidate(
            "Gradient Boosting",
            GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42),
            feature_set="engineered",
            log_target=True,
        ),
        Candidate(
            "HistGradientBoosting",
            HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, l2_regularization=10, random_state=42),
            feature_set="engineered",
        ),
        Candidate(
            "HistGradientBoosting",
            HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, l2_regularization=10, random_state=42),
            feature_set="engineered",
            log_target=True,
        ),
    ]


def build_candidate(candidate: Candidate):
    preprocessor = build_engineered_preprocessor() if candidate.feature_set == "engineered" else build_base_preprocessor()
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", candidate.estimator)])
    if not candidate.log_target:
        return pipeline
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def evaluate(candidate: Candidate, x_train, y_train, x_test, y_test, cv) -> dict:
    model = build_candidate(candidate)
    started = time.perf_counter()
    scores = cross_validate(model, x_train, y_train, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise")
    cv_seconds = time.perf_counter() - started

    fit_started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    prediction = model.predict(x_test)
    predict_ms_per_row = (time.perf_counter() - predict_started) * 1000 / len(x_test)

    result = {
        "model": candidate.name,
        "feature_set": candidate.feature_set,
        "target": "log1p" if candidate.log_target else "raw",
        "cv_mae_mean": float(-scores["test_mae"].mean()),
        "cv_mae_std": float(scores["test_mae"].std()),
        "cv_rmse_mean": float(-scores["test_rmse"].mean()),
        "cv_rmse_std": float(scores["test_rmse"].std()),
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std()),
        "cv_folds": {
            "mae": (-scores["test_mae"]).tolist(),
            "rmse": (-scores["test_rmse"]).tolist(),
            "r2": scores["test_r2"].tolist(),
        },
        "legacy_test_mae": float(mean_absolute_error(y_test, prediction)),
        "legacy_test_rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
        "legacy_test_r2": float(r2_score(y_test, prediction)),
        "cv_seconds": cv_seconds,
        "fit_seconds": fit_seconds,
        "predict_ms_per_row": predict_ms_per_row,
    }
    return result


def _markdown_table(results: list[dict]) -> str:
    rows = []
    for result in sorted(results, key=lambda item: item["cv_mae_mean"]):
        rows.append(
            "| {model} | {feature_set} | {target} | ${cv_mae_mean:,.0f} ± ${cv_mae_std:,.0f} | "
            "${cv_rmse_mean:,.0f} ± ${cv_rmse_std:,.0f} | {cv_r2_mean:.3f} ± {cv_r2_std:.3f} | "
            "${legacy_test_mae:,.0f} | ${legacy_test_rmse:,.0f} | {legacy_test_r2:.3f} |".format(**result)
        )
    return "\n".join(rows)


def render_report(results: list[dict], best_mae: dict, best_rmse: dict) -> str:
    return f"""# Seattle Model Benchmark

This benchmark uses five-fold shuffled KFold cross-validation with `random_state=42` on the cleaned training set. Models are ranked and shortlisted from CV only. The old 504-row test file is reported as a **legacy holdout** because previous project phases examined it repeatedly; it is not used for selection.

| Model | Features | Target | CV MAE | CV RMSE | CV R² | Legacy test MAE | Legacy test RMSE | Legacy test R² |
|---|---|---|---:|---:|---:|---:|---:|---:|
{_markdown_table(results)}

## CV shortlists

- Lowest CV MAE: **{best_mae['model']}**, {best_mae['feature_set']} features, {best_mae['target']} target — ${best_mae['cv_mae_mean']:,.0f} ± ${best_mae['cv_mae_std']:,.0f}.
- Lowest CV RMSE: **{best_rmse['model']}**, {best_rmse['feature_set']} features, {best_rmse['target']} target — ${best_rmse['cv_rmse_mean']:,.0f} ± ${best_rmse['cv_rmse_std']:,.0f}.

No final model is selected by this report. The strongest stable candidates proceed to bounded tuning and segment-level error analysis. The large RMSE standard deviations are reported rather than hidden because the luxury tail is concentrated unevenly across folds.
"""


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    validate_seattle_dataframe(train, stage="cleaned")
    validate_seattle_dataframe(test, stage="cleaned")
    x_train, y_train = train[FEATURES], train[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for candidate in candidates():
        label = f"{candidate.name} / {candidate.feature_set} / {'log1p' if candidate.log_target else 'raw'}"
        print(f"Evaluating {label}")
        results.append(evaluate(candidate, x_train, y_train, x_test, y_test, cv))

    best_mae = min(results, key=lambda item: item["cv_mae_mean"])
    best_rmse = min(results, key=lambda item: item["cv_rmse_mean"])
    payload = {
        "selection_policy": "CV only; legacy test metrics are not used for selection",
        "cv": {"type": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "train_rows": len(train),
        "legacy_test_rows": len(test),
        "results": results,
        "best_cv_mae": best_mae,
        "best_cv_rmse": best_rmse,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    pd.DataFrame([{key: value for key, value in result.items() if key != "cv_folds"} for result in results]).to_csv(CSV_PATH, index=False)
    REPORT_PATH.write_text(render_report(results, best_mae, best_rmse), encoding="utf-8", newline="\n")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
