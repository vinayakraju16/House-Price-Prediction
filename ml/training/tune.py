"""Bounded nested tuning for the strongest Seattle benchmark candidate.

Run from the repository root:
    python -m ml.training.tune
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from ml.data.validation import validate_seattle_dataframe
from ml.features.build_features import build_engineered_preprocessor

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "Seattle" / "train_cleaned.csv"
TEST_PATH = ROOT / "data" / "Seattle" / "test_cleaned.csv"
RESULT_PATH = ROOT / "reports" / "tuning_results.json"
CANDIDATE_PATH = ROOT / "reports" / "tuned_hist_gradient_boosting.joblib"
FEATURES = ["beds", "baths", "size", "lot_size", "zip_code"]
TARGET = "price"
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
PARAMETERS = {
    "regressor__regressor__learning_rate": [0.025, 0.05, 0.075, 0.1],
    "regressor__regressor__max_iter": [100, 150, 200, 300],
    "regressor__regressor__max_leaf_nodes": [7, 15, 31],
    "regressor__regressor__min_samples_leaf": [10, 20, 30, 40],
    "regressor__regressor__l2_regularization": [0.0, 1.0, 10.0, 50.0, 100.0],
}


def build_candidate():
    pipeline = Pipeline([
        ("preprocessor", build_engineered_preprocessor()),
        ("regressor", HistGradientBoostingRegressor(random_state=42)),
    ])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def build_search(*, folds: int) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator=build_candidate(),
        param_distributions=PARAMETERS,
        n_iter=12,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=folds, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )


def _metric_summary(scores, name: str, *, negate: bool = False) -> dict:
    values = np.asarray(scores[f"test_{name}"])
    if negate:
        values = -values
    return {"mean": float(values.mean()), "std": float(values.std()), "folds": values.tolist()}


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    validate_seattle_dataframe(train, stage="cleaned")
    validate_seattle_dataframe(test, stage="cleaned")
    x_train, y_train = train[FEATURES], train[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]

    print("Running five-fold outer / three-fold inner nested evaluation...")
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    nested_scores = cross_validate(
        build_search(folds=3), x_train, y_train, cv=outer_cv, scoring=SCORING, n_jobs=1, error_score="raise"
    )

    print("Fitting bounded search on all training rows...")
    search = build_search(folds=5)
    search.fit(x_train, y_train)
    prediction = search.best_estimator_.predict(x_test)
    joblib.dump(search.best_estimator_, CANDIDATE_PATH)

    payload = {
        "selection_metric": "mean cross-validated MAE in original dollars",
        "search": {
            "type": "RandomizedSearchCV",
            "iterations": 12,
            "inner_folds_for_nested_evaluation": 3,
            "outer_folds": 5,
            "final_search_folds": 5,
            "random_state": 42,
            "parameter_space": PARAMETERS,
        },
        "nested_cv": {
            "mae": _metric_summary(nested_scores, "mae", negate=True),
            "rmse": _metric_summary(nested_scores, "rmse", negate=True),
            "r2": _metric_summary(nested_scores, "r2"),
        },
        "best_params": search.best_params_,
        "best_search_cv_mae": float(-search.best_score_),
        "legacy_test": {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
            "r2": float(r2_score(y_test, prediction)),
        },
        "candidate_artifact": str(CANDIDATE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "deployment_status": "candidate_only",
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"nested_cv": payload["nested_cv"], "best_params": payload["best_params"], "legacy_test": payload["legacy_test"]}, indent=2))


if __name__ == "__main__":
    main()
