"""Bounded grouped-CV tuning for the winning King County model family."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from ml.data.king_county import MODEL_FEATURES
from ml.training.benchmark_king_county import Candidate, build_model, temporal_group_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "king-county" / "sales.csv"
RESULT_PATH = ROOT / "reports" / "king_county_tuning.json"
MODEL_PATH = ROOT / "reports" / "king_county_tuned_model.joblib"


def main() -> None:
    data = pd.read_csv(DATA_PATH, dtype={"zip_code": "string", "heat_system": "string",
                                        "property_type": "string"})
    development, holdout = temporal_group_split(data)
    candidate = Candidate(
        "HistGradientBoosting rich log",
        HistGradientBoostingRegressor(random_state=42),
        rich=True,
        log_target=True,
    )
    model = build_model(candidate)
    parameter_distributions = {
        "regressor__regressor__learning_rate": [0.035, 0.05, 0.07, 0.1],
        "regressor__regressor__max_iter": [200, 300, 400],
        "regressor__regressor__max_leaf_nodes": [15, 31, 63],
        "regressor__regressor__min_samples_leaf": [20, 30, 50, 80],
        "regressor__regressor__l2_regularization": [1.0, 5.0, 10.0, 20.0, 40.0],
        "regressor__regressor__max_bins": [127, 255],
    }
    search = RandomizedSearchCV(
        model,
        parameter_distributions,
        n_iter=12,
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=5),
        random_state=42,
        n_jobs=1,
        verbose=2,
        refit=True,
        error_score="raise",
    )
    search.fit(
        development[MODEL_FEATURES],
        development["price"],
        groups=development["parcel_id"],
    )
    prediction = search.best_estimator_.predict(holdout[MODEL_FEATURES])
    residuals = holdout["price"].to_numpy() - prediction
    payload = {
        "selection_policy": "12-iteration RandomizedSearchCV using five-fold GroupKFold RMSE",
        "best_params": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
        "holdout": {
            "rows": len(holdout),
            "mae": float(mean_absolute_error(holdout["price"], prediction)),
            "rmse": float(mean_squared_error(holdout["price"], prediction) ** 0.5),
            "r2": float(r2_score(holdout["price"], prediction)),
        },
        "prediction_interval": {
            "method": "2025-2026 temporal-holdout residual quantiles",
            "nominal_coverage": 0.9,
            "lower_residual_quantile": float(np.quantile(residuals, 0.05)),
            "upper_residual_quantile": float(np.quantile(residuals, 0.95)),
            "empirical_coverage": float(np.mean(
                (holdout["price"].to_numpy() >= prediction + np.quantile(residuals, 0.05))
                & (holdout["price"].to_numpy() <= prediction + np.quantile(residuals, 0.95))
            )),
        },
        "trials": [
            {
                "rank": int(search.cv_results_["rank_test_score"][index]),
                "mean_cv_rmse": float(-search.cv_results_["mean_test_score"][index]),
                "std_cv_rmse": float(search.cv_results_["std_test_score"][index]),
                "params": search.cv_results_["params"][index],
            }
            for index in range(len(search.cv_results_["params"]))
        ],
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    joblib.dump(search.best_estimator_, MODEL_PATH)
    print(json.dumps({key: value for key, value in payload.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
