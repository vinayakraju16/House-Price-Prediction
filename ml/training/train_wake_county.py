"""Benchmark, tune, evaluate and save the Wake County, NC sale-price model.

    python -m ml.data.wake_county            # prepare data/processed/wake-county/sales.csv
    python -m ml.training.train_wake_county  # ablation + tuning + final model

Split: sales from ``HOLDOUT_START`` onward form an untouched temporal holdout; parcels
that appear in it are removed from development data. Models are selected and tuned on a
*temporal* validation inside development (train before ``VALIDATION_START``, validate on
the months after it), because shuffled cross-validation mixes months and rewards features
that only describe "when" a sale happened. Five-fold GroupKFold CV is still reported to
show that gap. The holdout is scored once per model and never used for choices.

The ablation answers "which new data actually helps?" by adding one source at a time:
property facts -> Census income -> Zillow home values -> Realtor.com inventory ->
mortgage rates, and a variant that predicts price relative to the ZIP's Zillow index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.pipeline import Pipeline

from ml.data.market_features import (
    ACS_FEATURES,
    INVENTORY_FEATURES,
    MARKET_DIR,
    RATE_FEATURES,
    ZHVI_FEATURES,
    MarketData,
    add_market_features,
    latest_snapshot,
)
from ml.data.wake_county import MODEL_FEATURES
from ml.features.build_features import build_property_market_preprocessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from model_runtime.market import MarketRelativeRegressor  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "wake-county" / "sales.csv"
REPORT_DIR = ROOT / "reports"
RESULT_PATH = REPORT_DIR / "wake_county_model_results.json"
REPORT_PATH = REPORT_DIR / "WAKE_COUNTY_MODEL.md"
MODEL_DIR = ROOT / "backend" / "mlmodels" / "versions"
VERSION = "wake-county-1.0.0"
HOLDOUT_START = "2026-04-01"
VALIDATION_START = "2026-01-01"
CATEGORICAL = ["zip_code", "heat_system", "exterior_wall", "property_type"]
NUMERIC = [name for name in MODEL_FEATURES if name not in CATEGORICAL]
CALENDAR = ["sale_year", "sale_month"]
DEFAULT_PARAMS = {"max_iter": 300, "max_leaf_nodes": 31, "learning_rate": 0.06,
                  "l2_regularization": 10.0, "min_samples_leaf": 30, "random_state": 42}
SEARCH_SPACE = {
    "learning_rate": [0.035, 0.05, 0.07, 0.1],
    "max_iter": [200, 300, 450, 600],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 30, 50, 80],
    "l2_regularization": [1.0, 5.0, 10.0, 20.0, 40.0],
    "max_bins": [127, 255],
}


@dataclass(frozen=True)
class Variant:
    name: str
    market: tuple[str, ...] = ()
    relative: bool = False
    drop: tuple[str, ...] = field(default_factory=tuple)
    baseline: bool = False

    @property
    def columns(self) -> list[str]:
        return [*NUMERIC, *CATEGORICAL, *self.market]


ALL_MARKET = (*ZHVI_FEATURES, *INVENTORY_FEATURES, *RATE_FEATURES, *ACS_FEATURES)
NO_CALENDAR = tuple(CALENDAR)
VARIANTS = [
    Variant("Median price (baseline)", baseline=True),
    Variant("Property facts only"),
    Variant("+ Census ZIP income", market=tuple(ACS_FEATURES)),
    Variant("+ Zillow ZHVI", market=tuple(ZHVI_FEATURES)),
    Variant("+ ZHVI + inventory", market=(*ZHVI_FEATURES, *INVENTORY_FEATURES)),
    Variant("+ all market data", market=ALL_MARKET),
    Variant("All market data, no calendar", market=ALL_MARKET, drop=NO_CALENDAR),
    Variant("ZHVI-relative: property + income", market=("zhvi", *ACS_FEATURES),
            relative=True, drop=NO_CALENDAR),
    Variant("ZHVI-relative: + ZHVI momentum", market=(*ZHVI_FEATURES, *ACS_FEATURES),
            relative=True, drop=NO_CALENDAR),
    Variant("ZHVI-relative: all market data", market=ALL_MARKET, relative=True,
            drop=NO_CALENDAR),
    Variant("ZHVI-relative: all market + calendar", market=ALL_MARKET, relative=True),
]


def build_model(variant: Variant, params: dict | None = None):
    if variant.baseline:
        return DummyRegressor(strategy="median")
    pipeline = Pipeline([
        ("preprocessor", build_property_market_preprocessor(NUMERIC, CATEGORICAL,
                                                            variant.market, variant.drop)),
        ("regressor", HistGradientBoostingRegressor(**{**DEFAULT_PARAMS, **(params or {})})),
    ])
    if variant.relative:
        return MarketRelativeRegressor(pipeline)
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def metrics(actual, prediction) -> dict:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    ape = np.abs(prediction - actual) / actual
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "median_ape": float(np.median(ape)),
        "within_10pct": float(np.mean(ape <= 0.10)),
    }


def temporal_group_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = data.loc[data["sale_date"] >= HOLDOUT_START].copy()
    development = data.loc[
        (data["sale_date"] < HOLDOUT_START) & ~data["parcel_id"].isin(set(holdout["parcel_id"]))
    ].copy()
    if development.empty or holdout.empty:
        raise ValueError("Need both development rows and holdout rows")
    return development, holdout


def temporal_validation_split(development: pd.DataFrame):
    validation = development.loc[development["sale_date"] >= VALIDATION_START]
    train = development.loc[(development["sale_date"] < VALIDATION_START)
                            & ~development["parcel_id"].isin(set(validation["parcel_id"]))]
    return train, validation


def validate(variant: Variant, development: pd.DataFrame, params=None) -> dict:
    train, validation = temporal_validation_split(development)
    model = build_model(variant, params).fit(train[variant.columns], train["price"])
    return metrics(validation["price"], model.predict(validation[variant.columns]))


def cross_validate(variant: Variant, development: pd.DataFrame, params=None) -> dict:
    folds = []
    for train_index, test_index in GroupKFold(n_splits=5).split(
        development, groups=development["parcel_id"]
    ):
        train, test = development.iloc[train_index], development.iloc[test_index]
        model = build_model(variant, params).fit(train[variant.columns], train["price"])
        folds.append(metrics(test["price"], model.predict(test[variant.columns])))
    return {key: float(np.mean([fold[key] for fold in folds])) for key in folds[0]
            if key != "rows"}


def load_dataset(market_dir: Path = MARKET_DIR) -> tuple[pd.DataFrame, MarketData]:
    data = pd.read_csv(DATA_PATH, dtype={"parcel_id": str, "zip_code": str, "heat_system": str,
                                        "exterior_wall": str, "property_type": str})
    market = MarketData.load(market_dir, zip_codes=data["zip_code"].unique())
    return add_market_features(data, market), market


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(market_dir: Path = MARKET_DIR, tune_trials: int = 12, save: bool = True) -> dict:
    data, market = load_dataset(market_dir)
    development, holdout = temporal_group_split(data)
    print(f"development {len(development):,} rows, holdout {len(holdout):,} rows", flush=True)

    ablation = []
    for variant in VARIANTS:
        started = time.perf_counter()
        cv = cross_validate(variant, development)
        valid = validate(variant, development)
        model = build_model(variant).fit(development[variant.columns], development["price"])
        test = metrics(holdout["price"], model.predict(holdout[variant.columns]))
        row = {"model": variant.name, "cv": cv, "validation": valid, "holdout": test,
               "seconds": round(time.perf_counter() - started, 1)}
        ablation.append(row)
        print(f"{variant.name:40s} CV MAE ${cv['mae']:>8,.0f}  valid MAE ${valid['mae']:>8,.0f}"
              f"  holdout MAE ${test['mae']:>8,.0f}  median APE {test['median_ape']:.2%}",
              flush=True)

    # Select on the temporal validation only; the holdout stays untouched.
    candidates = [(row, variant) for row, variant in zip(ablation, VARIANTS, strict=True)
                  if not variant.baseline]
    best_row, best_variant = min(candidates, key=lambda item: item[0]["validation"]["mae"])
    print(f"selected by temporal-validation MAE: {best_variant.name}", flush=True)

    trials = []
    for params in ParameterSampler(SEARCH_SPACE, n_iter=tune_trials, random_state=42):
        valid = validate(best_variant, development, params)
        trials.append({"params": params, "validation": valid})
        print(f"  trial valid MAE ${valid['mae']:,.0f} {params}", flush=True)
    best_trial = min(trials, key=lambda trial: trial["validation"]["mae"])
    if best_trial["validation"]["mae"] > best_row["validation"]["mae"]:
        best_trial = {"params": {}, "validation": best_row["validation"]}
    params = best_trial["params"]

    model = build_model(best_variant, params).fit(development[best_variant.columns],
                                                  development["price"])
    prediction = model.predict(holdout[best_variant.columns])
    log_residuals = np.log(holdout["price"].to_numpy()) - np.log(prediction)
    low, high = np.quantile(log_residuals, [0.05, 0.95])
    covered = (log_residuals >= low) & (log_residuals <= high)
    frame = holdout.assign(prediction=prediction)
    frame["segment"] = pd.cut(frame["price"], bins=[0, 300_000, 500_000, 800_000, np.inf],
                              labels=["Below $300K", "$300K-$500K", "$500K-$800K", "$800K+"])
    importance_sample = holdout.sample(n=min(3000, len(holdout)), random_state=42)
    importance = permutation_importance(
        model, importance_sample[best_variant.columns], importance_sample["price"],
        scoring="neg_mean_absolute_error", n_repeats=3, random_state=42,
    )
    evaluation = {
        "overall": metrics(holdout["price"], prediction),
        "price_segments": {str(name): metrics(group["price"], group["prediction"])
                           for name, group in frame.groupby("segment", observed=True)},
        "new_construction": {
            label: metrics(group["price"], group["prediction"])
            for label, group in frame.groupby(
                np.where(frame["sale_year"] - frame["year_built"] <= 1, "new", "resale"))
        },
        "zip_segments": {str(name): metrics(group["price"], group["prediction"])
                         for name, group in frame.groupby("zip_code") if len(group) >= 50},
        "feature_importance": sorted(
            [{"feature": name, "mae_increase": float(mean), "std": float(std)}
             for name, mean, std in zip(best_variant.columns, importance.importances_mean,
                                        importance.importances_std, strict=True)],
            key=lambda item: item["mae_increase"], reverse=True),
        "prediction_interval": {
            "method": "holdout log-residual quantiles (percentage band around the estimate)",
            "nominal_coverage": 0.9,
            "lower_ratio": float(np.exp(low)), "upper_ratio": float(np.exp(high)),
            "empirical_coverage": float(covered.mean()),
        },
    }
    results = {
        "split": {"holdout_start": HOLDOUT_START, "development_rows": len(development),
                  "holdout_rows": len(holdout),
                  "development_range": [development["sale_date"].min(),
                                        development["sale_date"].max()],
                  "holdout_range": [holdout["sale_date"].min(), holdout["sale_date"].max()]},
        "ablation": ablation,
        "selected_variant": best_variant.name,
        "selection": "temporal validation MAE (train before "
                     f"{VALIDATION_START}, validate {VALIDATION_START} to {HOLDOUT_START})",
        "tuning": {"trials": trials, "best_params": params,
                   "best_validation": best_trial["validation"]},
        "evaluation": evaluation,
        "runtime": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__},
    }
    if save:
        save_outputs(results, best_variant, params, data, market)
    return results


def save_outputs(results, variant, params, data, market) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8", newline="\n")
    REPORT_PATH.write_text(render_report(results), encoding="utf-8", newline="\n")

    save_model(results, variant, params, data, market)


def _defaults(data: pd.DataFrame) -> dict:
    defaults = {}
    for name in [*NUMERIC, *CATEGORICAL]:
        if name == "year_renovated":
            defaults[name] = None
        elif name in CATEGORICAL:
            defaults[name] = str(data[name].mode(dropna=True).iloc[0])
        else:
            defaults[name] = float(data[name].median())
    return defaults


def _ranges(data: pd.DataFrame) -> dict:
    ranges = {}
    for name in NUMERIC:
        values = pd.to_numeric(data[name], errors="coerce").dropna()
        if not values.empty:
            ranges[name] = {"minimum": float(values.min()), "p05": float(values.quantile(0.05)),
                            "p95": float(values.quantile(0.95)), "maximum": float(values.max())}
    return ranges


def save_model(results, variant, params, data, market) -> None:
    final = build_model(variant, params).fit(data[variant.columns], data["price"])
    snapshot = latest_snapshot(market, data["zip_code"].unique())
    version_dir = MODEL_DIR / VERSION
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.joblib"
    comparable_columns = ["baths", "size", "lot_size", "zip_code", "price", "sale_date",
                          "year_built", "property_type"]
    joblib.dump({"best_model": final, "feature_names": variant.columns,
                 "market_snapshot": snapshot.to_dict(orient="records"),
                 "comparables": data.sort_values("sale_date")[comparable_columns]
                 .tail(15_000).to_dict(orient="records"),
                 "model_version": VERSION}, model_path, compress=3)
    metadata = {
        "model_version": VERSION,
        "status": "experimental",
        "market": "Wake County, NC",
        "variant": variant.name,
        "features": variant.columns,
        "market_features": list(variant.market),
        "excluded_from_model": list(variant.drop),
        "required_features": ["baths", "size", "lot_size", "zip_code"],
        "feature_defaults": _defaults(data),
        "typical_feature_ranges": _ranges(data),
        "zip_codes": sorted(data["zip_code"].unique().tolist()),
        "target": "log(price / ZHVI)" if variant.relative else "log1p(price)",
        "hyperparameters": {**DEFAULT_PARAMS, **params},
        "training_rows": len(data),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "metrics": {**results["evaluation"]["overall"],
                    "evaluation_label": f"temporal holdout from {HOLDOUT_START}"},
        "prediction_interval": results["evaluation"]["prediction_interval"],
        "market_snapshot_month_index": int(snapshot["zhvi_month_index"].max())
        if "zhvi_month_index" in snapshot else None,
        "runtime": results["runtime"],
        "artifact": {"relative_path": model_path.relative_to(ROOT).as_posix(),
                     "sha256": _sha256(model_path)},
        "limitations": [
            "Wake County publishes no bedroom count; the model has no beds input.",
            "Only two years of qualified sales are published, so long-run cycles are unseen.",
            "Market features are ZIP-level; they do not describe individual streets.",
        ],
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2),
                                                encoding="utf-8", newline="\n")
    print(f"saved {model_path.relative_to(ROOT)}")


def _money(value: float) -> str:
    return f"${value:,.0f}"


def render_report(results: dict) -> str:
    rows = "\n".join(
        f"| {row['model']} | {_money(row['cv']['mae'])} | {_money(row['validation']['mae'])} | "
        f"{_money(row['holdout']['mae'])} | {_money(row['holdout']['rmse'])} | "
        f"{row['holdout']['r2']:.3f} | {row['holdout']['median_ape']:.1%} | "
        f"{row['holdout']['within_10pct']:.0%} |"
        for row in results["ablation"]
    )
    evaluation = results["evaluation"]
    overall = evaluation["overall"]
    segments = "\n".join(
        f"| {name} | {values['rows']:,} | {_money(values['mae'])} | {values['median_ape']:.1%} |"
        for name, values in evaluation["price_segments"].items()
    )
    construction = "\n".join(
        f"| {name} | {values['rows']:,} | {_money(values['mae'])} | {values['median_ape']:.1%} |"
        for name, values in evaluation["new_construction"].items()
    )
    importance = "\n".join(
        f"| {item['feature']} | {_money(item['mae_increase'])} |"
        for item in evaluation["feature_importance"][:12]
    )
    interval = evaluation["prediction_interval"]
    split = results["split"]
    selected = next(row for row in results["ablation"]
                    if row["model"] == results["selected_variant"])
    return f"""# Wake County, NC Model

Development sales {split['development_range'][0]} to {split['development_range'][1]}
({split['development_rows']:,} rows); temporal holdout {split['holdout_range'][0]} to
{split['holdout_range'][1]} ({split['holdout_rows']:,} rows). Holdout parcels are excluded
from development.

Models are chosen on a temporal validation inside development ({results['selection']}).
Shuffled five-fold GroupKFold CV is shown for comparison only: it mixes months, so it
rewards features that merely encode *when* a sale happened, which then fail on later sales.

## Which data helps (feature ablation)

| Model | Shuffled CV MAE | Temporal validation MAE | Holdout MAE | Holdout RMSE | Holdout R² | Holdout median error | Within 10% |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

Selected: **{results['selected_variant']}**, then tuned with
{len(results['tuning']['trials'])} random trials on the same temporal validation
(untuned holdout MAE {_money(selected['holdout']['mae'])}; small tuning gains on validation
do not always carry over to later months).

## Final model on the holdout

MAE **{_money(overall['mae'])}**, RMSE **{_money(overall['rmse'])}**, R² **{overall['r2']:.3f}**,
median error **{overall['median_ape']:.1%}**, within 10% of the sale price
**{overall['within_10pct']:.0%}** of the time.

| Price band | Rows | MAE | Median error |
|---|---:|---:|---:|
{segments}

| Sale type | Rows | MAE | Median error |
|---|---:|---:|---:|
{construction}

90% range: estimate × {interval['lower_ratio']:.2f} to × {interval['upper_ratio']:.2f}
(holdout coverage {interval['empirical_coverage']:.0%}).

## Most important inputs (holdout permutation importance, MAE increase)

| Feature | MAE increase when shuffled |
|---|---:|
{importance}

Runtime: Python {results['runtime']['python']}, scikit-learn {results['runtime']['scikit_learn']},
pandas {results['runtime']['pandas']}.
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--market-dir", type=Path, default=MARKET_DIR)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--final-only", action="store_true",
                        help="refit and save the final model from the saved results JSON")
    args = parser.parse_args(argv)
    if args.final_only:
        results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        variant = next(v for v in VARIANTS if v.name == results["selected_variant"])
        data, market = load_dataset(args.market_dir)
        save_model(results, variant, results["tuning"]["best_params"], data, market)
    else:
        run(args.market_dir, args.trials)


if __name__ == "__main__":
    main()
