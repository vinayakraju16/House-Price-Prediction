"""Out-of-fold and legacy-holdout error analysis for the tuned Seattle candidate.

Run from the repository root:
    python -m ml.evaluation.error_analysis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from ml.data.validation import validate_seattle_dataframe

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
TRAIN_PATH = ROOT / "data" / "Seattle" / "train_cleaned.csv"
TEST_PATH = ROOT / "data" / "Seattle" / "test_cleaned.csv"
CANDIDATE_PATH = ROOT / "reports" / "tuned_hist_gradient_boosting.joblib"
TUNING_PATH = ROOT / "reports" / "tuning_results.json"
SUMMARY_PATH = ROOT / "reports" / "model_evaluation.json"
REPORT_PATH = ROOT / "reports" / "MODEL_EVALUATION.md"
FIGURE_DIR = ROOT / "reports" / "figures"
FEATURES = ["beds", "baths", "size", "lot_size", "zip_code"]
TARGET = "price"


def metrics(actual, predicted) -> dict:
    return {
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "median_absolute_error": float(np.median(np.abs(np.asarray(actual) - np.asarray(predicted)))),
    }


def segment_metrics(frame: pd.DataFrame, segment: pd.Series, prediction: np.ndarray, *, minimum_rows: int = 2) -> list[dict]:
    working = pd.DataFrame({"segment": segment.astype(str), "actual": frame[TARGET], "prediction": prediction})
    results = []
    for label, group in working.groupby("segment", sort=False):
        if len(group) < minimum_rows:
            continue
        results.append({"segment": label, **metrics(group["actual"], group["prediction"])})
    return results


def _save_figures(actual: pd.Series, prediction: np.ndarray, price_segments: list[dict]) -> None:
    residual = actual.to_numpy() - prediction
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(prediction, residual, alpha=0.25, s=12)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Out-of-fold prediction ($)")
    axes[0].set_ylabel("Actual - prediction ($)")
    axes[0].set_title("Residuals vs predictions")
    axes[0].ticklabel_format(style="plain")
    axes[1].hist(residual, bins=60, color="#315f72", edgecolor="white")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual ($)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "residual_analysis.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(actual, prediction, alpha=0.25, s=12)
    lower = max(1, min(float(actual.min()), float(prediction.min())))
    upper = max(float(actual.max()), float(prediction.max()))
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Actual price ($, log scale)")
    axis.set_ylabel("OOF prediction ($, log scale)")
    axis.set_title("Actual vs out-of-fold prediction")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "actual_vs_prediction.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar([item["segment"] for item in price_segments], [item["mae"] for item in price_segments], color="#c87842")
    axis.set_ylabel("MAE ($)")
    axis.set_title("Out-of-fold MAE by actual price band")
    axis.tick_params(axis="x", rotation=20)
    axis.ticklabel_format(axis="y", style="plain")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "error_by_price_band.png", dpi=160)
    plt.close(fig)


def _table(items: list[dict]) -> str:
    rows = []
    for item in items:
        r2 = "n/a" if item["r2"] is None else f"{item['r2']:.3f}"
        rows.append(
            f"| {item['segment']} | {item['rows']:,} | ${item['mae']:,.0f} | ${item['rmse']:,.0f} | {r2} |"
        )
    return "\n".join(rows)


def render_report(summary: dict) -> str:
    nested = summary["nested_cv"]
    oof = summary["fixed_candidate_oof"]
    legacy = summary["legacy_test"]
    interval = summary["prediction_interval"]
    largest_rows = "\n".join(
        f"| ${item['actual']:,.0f} | ${item['prediction']:,.0f} | ${item['absolute_error']:,.0f} | {item['beds']:g} | {item['baths']:g} | {item['size']:,.0f} | {int(item['zip_code'])} |"
        for item in summary["largest_oof_errors"]
    )
    return f"""# Seattle Model Evaluation

Candidate: tuned HistGradientBoosting with inference-safe engineered features and a `log1p` target. Hyperparameters were selected with a bounded RandomizedSearchCV. The primary generalization estimate is five-fold outer nested CV; the repeatedly examined test file is labeled as a legacy holdout.

## Overall results

| Evaluation | MAE | RMSE | R² |
|---|---:|---:|---:|
| Median dummy CV | $384,376 | $876,816 | -0.045 |
| Active Ridge CV | $243,888 | $693,739 | 0.434 |
| Tuned candidate nested CV | ${nested['mae']['mean']:,.0f} ± ${nested['mae']['std']:,.0f} | ${nested['rmse']['mean']:,.0f} ± ${nested['rmse']['std']:,.0f} | {nested['r2']['mean']:.3f} ± {nested['r2']['std']:.3f} |
| Tuned fixed-parameter OOF | ${oof['mae']:,.0f} | ${oof['rmse']:,.0f} | {oof['r2']:.3f} |
| Tuned candidate legacy holdout | ${legacy['mae']:,.0f} | ${legacy['rmse']:,.0f} | {legacy['r2']:.3f} |

The tuned candidate reduces nested CV MAE by about {summary['improvement_over_ridge']['mae_percent']:.1f}% and nested CV RMSE by about {summary['improvement_over_ridge']['rmse_percent']:.1f}% relative to the active Ridge benchmark. It also clearly outperforms the dummy baseline. RMSE remains unstable because two validation folds contain the most extreme luxury observations.

## Prediction interval calibration

The API interval uses asymmetric 5th and 95th percentiles of five-fold out-of-fold residuals. The measured training OOF coverage is {interval['observed_oof_coverage']:.1%} for a nominal 90% interval. The residual adjustments are ${interval['lower_residual_quantile']:,.0f} and +${interval['upper_residual_quantile']:,.0f}. This empirical interval is more defensible than an arbitrary percentage or symmetric RMSE band, but it is not a guarantee and can under-cover distribution shifts or sparse luxury homes.

![Actual vs prediction](figures/actual_vs_prediction.png)

![Residual analysis](figures/residual_analysis.png)

## Error by actual price band

| Price band | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
{_table(summary['segments']['price_band'])}

![Error by price band](figures/error_by_price_band.png)

## Error by property size

| Size band | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
{_table(summary['segments']['size_band'])}

## Error by bedrooms

| Bedroom group | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
{_table(summary['segments']['bedrooms'])}

## Error by ZIP

ZIP rows with fewer than ten training observations are omitted from this table because their metrics are too unstable.

| ZIP | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
{_table(summary['segments']['zip_code'])}

## Largest out-of-fold errors

| Actual | Prediction | Absolute error | Beds | Baths | Size | ZIP |
|---:|---:|---:|---:|---:|---:|---:|
{largest_rows}

## Selection decision

The tuned HistGradientBoosting candidate is selected for the next deployable version because it wins both nested CV MAE and RMSE against the current Ridge model, improves typical-market errors substantially, remains fast for single-row inference, and uses only features reproducible by the API. The selection does not rely on the legacy test ranking.

Limitations remain explicit: only five source fields are available, the luxury segment is sparse, the test set is no longer pristine, and no current-sale or verified comparable data exists. The next model version should therefore remain experimental rather than being described as an appraisal.
"""


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    validate_seattle_dataframe(train, stage="cleaned")
    validate_seattle_dataframe(test, stage="cleaned")
    model = joblib.load(CANDIDATE_PATH)
    tuning = json.loads(TUNING_PATH.read_text(encoding="utf-8"))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_prediction = cross_val_predict(model, train[FEATURES], train[TARGET], cv=cv, n_jobs=1)
    legacy_prediction = model.predict(test[FEATURES])

    price_band = pd.cut(
        train[TARGET],
        bins=[0, 500_000, 1_000_000, 2_000_000, np.inf],
        labels=["Below $500K", "$500K-$1M", "$1M-$2M", "Above $2M"],
        include_lowest=True,
    )
    size_band = pd.cut(
        train["size"],
        bins=[0, 1000, 2000, 3500, np.inf],
        labels=["Below 1,000 sqft", "1,000-2,000 sqft", "2,000-3,500 sqft", "Above 3,500 sqft"],
        include_lowest=True,
    )
    bedroom_group = pd.cut(
        train["beds"], bins=[0, 2, 3, 4, np.inf], labels=["1-2", "3", "4", "5+"], include_lowest=True
    )

    absolute_error = np.abs(train[TARGET].to_numpy() - oof_prediction)
    residual = train[TARGET].to_numpy() - oof_prediction
    residual_lower, residual_upper = np.quantile(residual, [0.05, 0.95])
    empirical_coverage = np.mean(
        (train[TARGET].to_numpy() >= oof_prediction + residual_lower)
        & (train[TARGET].to_numpy() <= oof_prediction + residual_upper)
    )
    largest_indices = np.argsort(absolute_error)[-10:][::-1]
    largest = []
    for index in largest_indices:
        row = train.iloc[index]
        largest.append({
            "actual": float(row[TARGET]),
            "prediction": float(oof_prediction[index]),
            "absolute_error": float(absolute_error[index]),
            **{feature: float(row[feature]) for feature in FEATURES},
        })

    nested = tuning["nested_cv"]
    summary = {
        "model": "HistGradientBoostingRegressor",
        "target_transform": "log1p/expm1",
        "best_params": tuning["best_params"],
        "nested_cv": nested,
        "fixed_candidate_oof": metrics(train[TARGET], oof_prediction),
        "legacy_test": metrics(test[TARGET], legacy_prediction),
        "prediction_interval": {
            "method": "asymmetric empirical quantiles of five-fold out-of-fold residuals",
            "nominal_coverage": 0.90,
            "observed_oof_coverage": float(empirical_coverage),
            "lower_residual_quantile": float(residual_lower),
            "upper_residual_quantile": float(residual_upper),
        },
        "improvement_over_ridge": {
            "mae_percent": float((243_887.93 - nested["mae"]["mean"]) / 243_887.93 * 100),
            "rmse_percent": float((693_738.68 - nested["rmse"]["mean"]) / 693_738.68 * 100),
        },
        "segments": {
            "price_band": segment_metrics(train, price_band, oof_prediction),
            "size_band": segment_metrics(train, size_band, oof_prediction),
            "bedrooms": segment_metrics(train, bedroom_group, oof_prediction),
            "zip_code": segment_metrics(train, train["zip_code"], oof_prediction, minimum_rows=10),
        },
        "largest_oof_errors": largest,
    }
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _save_figures(train[TARGET], oof_prediction, summary["segments"]["price_band"])
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
