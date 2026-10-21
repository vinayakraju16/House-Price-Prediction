"""Does ZIP-level market data improve the King County (Seattle) model? Evaluation only.

Uses the deployed 3.0.0 split (development through 2024, holdout 2025-2026, holdout
parcels removed from development) and its tuned hyperparameters, so differences come
from the added data alone. Nothing is registered or exported.

    python -m ml.training.evaluate_king_county_market
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from ml.data.king_county import MODEL_FEATURES
from ml.data.market_features import (
    ACS_FEATURES,
    INVENTORY_FEATURES,
    MARKET_DIR,
    MARKET_FEATURES,
    RATE_FEATURES,
    ZHVI_FEATURES,
    MarketData,
    add_market_features,
)
from ml.features.build_features import (
    build_king_county_preprocessor,
    build_property_market_preprocessor,
)
from ml.training.benchmark_king_county import temporal_group_split
from ml.training.train_wake_county import metrics
from model_runtime.market import MarketRelativeRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "king-county" / "sales.csv"
TUNING_PATH = ROOT / "reports" / "king_county_tuning.json"
RESULT_PATH = ROOT / "reports" / "king_county_market_features.json"
CATEGORICAL = ["zip_code", "heat_system", "property_type"]
NUMERIC = [name for name in MODEL_FEATURES if name not in CATEGORICAL]
CALENDAR = ["sale_year", "sale_month"]


def tuned_params() -> dict:
    params = json.loads(TUNING_PATH.read_text(encoding="utf-8"))["best_params"]
    return {key.rsplit("__", 1)[-1]: value for key, value in params.items()} | {
        "random_state": 42}


def variants(params: dict) -> list[tuple[str, list[str], object]]:
    def generic(market=(), relative=False, drop=()):
        numeric = [name for name in NUMERIC if name not in drop]
        pipeline = Pipeline([
            ("preprocessor", build_property_market_preprocessor(numeric, CATEGORICAL, market)),
            ("regressor", HistGradientBoostingRegressor(**params)),
        ])
        columns = [*numeric, *CATEGORICAL, *market]
        if relative:
            return columns, MarketRelativeRegressor(pipeline)
        return columns, TransformedTargetRegressor(pipeline, func=np.log1p, inverse_func=np.expm1)

    production = TransformedTargetRegressor(
        Pipeline([("preprocessor", build_king_county_preprocessor()),
                  ("regressor", HistGradientBoostingRegressor(**params))]),
        func=np.log1p, inverse_func=np.expm1,
    )
    return [
        ("Deployed 3.0.0 recipe (refit here)", list(MODEL_FEATURES), production),
        ("Property facts only", *generic()),
        ("+ Zillow ZHVI", *generic(ZHVI_FEATURES)),
        ("+ ZHVI + inventory", *generic((*ZHVI_FEATURES, *INVENTORY_FEATURES))),
        ("+ all market data", *generic(MARKET_FEATURES)),
        ("All market data, ZHVI-relative target", *generic(MARKET_FEATURES, relative=True)),
        ("ZHVI-relative target, no calendar",
         *generic(MARKET_FEATURES, relative=True, drop=CALENDAR)),
        ("ZHVI-relative, ZHVI + income only, no calendar",
         *generic((*ZHVI_FEATURES, *ACS_FEATURES), relative=True, drop=CALENDAR)),
        ("+ rates only (calendar proxy check)", *generic(RATE_FEATURES)),
        ("+ inventory only", *generic(INVENTORY_FEATURES)),
    ]


def run(market_dir: Path = MARKET_DIR) -> dict:
    data = pd.read_csv(DATA_PATH, dtype={"parcel_id": str, "zip_code": str,
                                        "heat_system": str, "property_type": str})
    market = MarketData.load(market_dir, zip_codes=data["zip_code"].unique())
    data = add_market_features(data, market)
    development, holdout = temporal_group_split(data)
    coverage = {column: float(data[column].notna().mean()) for column in MARKET_FEATURES}
    results = []
    for name, columns, model in variants(tuned_params()):
        model.fit(development[columns], development["price"])
        prediction = model.predict(holdout[columns])
        by_year = {
            str(year): metrics(group["price"], prediction[holdout.index.get_indexer(group.index)])
            for year, group in holdout.groupby("sale_year")
        }
        row = {"model": name, "holdout": metrics(holdout["price"], prediction),
               "holdout_by_year": by_year}
        results.append(row)
        print(f"{name:50s} MAE ${row['holdout']['mae']:>9,.0f}  RMSE "
              f"${row['holdout']['rmse']:>9,.0f}  R2 {row['holdout']['r2']:.3f}  median APE "
              f"{row['holdout']['median_ape']:.2%}", flush=True)
    payload = {
        "split": "development through 2024; holdout 2025-2026; holdout parcels excluded",
        "development_rows": len(development), "holdout_rows": len(holdout),
        "market_feature_coverage": coverage, "hyperparameters": tuned_params(),
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--market-dir", type=Path, default=MARKET_DIR)
    run(parser.parse_args(argv).market_dir)


if __name__ == "__main__":
    main()
