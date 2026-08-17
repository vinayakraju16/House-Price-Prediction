"""Generate reproducible Seattle exploratory analysis and figures.

Run from the repository root:
    python -m ml.evaluation.eda
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data.validation import validate_seattle_dataframe

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Seattle" / "train_cleaned.csv"
REPORT_PATH = ROOT / "reports" / "EDA_REPORT.md"
SUMMARY_PATH = ROOT / "reports" / "eda_summary.json"
FIGURE_DIR = ROOT / "reports" / "figures"
FEATURES = ["beds", "baths", "size", "lot_size", "zip_code"]
TARGET = "price"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _save_price_distribution(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(data[TARGET], bins=50, color="#315f72", edgecolor="white")
    axes[0].set_title("Sale price distribution")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Listings")
    axes[0].ticklabel_format(axis="x", style="plain")
    axes[1].hist(np.log1p(data[TARGET]), bins=50, color="#c87842", edgecolor="white")
    axes[1].set_title("log1p(sale price) distribution")
    axes[1].set_xlabel("log1p(price)")
    axes[1].set_ylabel("Listings")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "price_distribution.png", dpi=160)
    plt.close(fig)


def _save_numeric_relationships(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, feature in zip(axes.flat, ["beds", "baths", "size", "lot_size"]):
        x = data[feature]
        if feature == "lot_size":
            visual_limit = float(x.quantile(0.99))
            visible = x <= visual_limit
            axis.scatter(x[visible], data.loc[visible, TARGET], alpha=0.25, s=12)
            axis.set_title("Lot size vs price (through p99)")
        else:
            axis.scatter(x, data[TARGET], alpha=0.25, s=12)
            axis.set_title(f"{feature.replace('_', ' ').title()} vs price")
        axis.set_xlabel(feature.replace("_", " "))
        axis.set_ylabel("Price ($)")
        axis.ticklabel_format(axis="y", style="plain")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_relationships.png", dpi=160)
    plt.close(fig)


def _save_zip_prices(data: pd.DataFrame) -> pd.DataFrame:
    summary = data.groupby("zip_code")[TARGET].agg(["count", "median", "mean"]).sort_values("median")
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.barh(summary.index.astype(str), summary["median"], color="#5f8066")
    axis.set_title("Median sale price by ZIP code")
    axis.set_xlabel("Median price ($)")
    axis.ticklabel_format(axis="x", style="plain")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "zip_median_price.png", dpi=160)
    plt.close(fig)
    return summary


def _save_correlations(data: pd.DataFrame) -> pd.DataFrame:
    correlation = data[[*FEATURES, TARGET]].corr(method="spearman")
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation.index)), correlation.index)
    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            axis.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("Spearman rank correlations")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlations.png", dpi=160)
    plt.close(fig)
    return correlation


def build_summary(data: pd.DataFrame) -> dict:
    validation = validate_seattle_dataframe(data, stage="cleaned")
    log_price = np.log1p(data[TARGET])
    q1, q3 = data[TARGET].quantile([0.25, 0.75])
    upper_outlier_boundary = float(q3 + 3 * (q3 - q1))
    return {
        "data_path": DATA_PATH.relative_to(ROOT).as_posix(),
        "rows": int(len(data)),
        "columns": list(data.columns),
        "validation": validation,
        "price": {
            "minimum": float(data[TARGET].min()),
            "median": float(data[TARGET].median()),
            "mean": float(data[TARGET].mean()),
            "maximum": float(data[TARGET].max()),
            "skew": float(data[TARGET].skew()),
            "log1p_skew": float(log_price.skew()),
            "three_iqr_upper_boundary": upper_outlier_boundary,
            "three_iqr_outlier_count": int((data[TARGET] > upper_outlier_boundary).sum()),
        },
        "missing_lot_indicator_count": int((data["lot_size"] == 0).sum()),
        "unique_zip_codes": int(data["zip_code"].nunique()),
    }


def render_report(summary: dict, zip_summary: pd.DataFrame, correlations: pd.DataFrame, data: pd.DataFrame) -> str:
    price = summary["price"]
    strongest = correlations[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    largest = data.nlargest(5, TARGET)[[TARGET, *FEATURES]]
    top_zips = zip_summary.tail(5).sort_values("median", ascending=False)
    low_zips = zip_summary.head(5)

    def zip_rows(frame: pd.DataFrame) -> str:
        return "\n".join(
            f"| {int(index)} | {int(row['count'])} | {_money(row['median'])} | {_money(row['mean'])} |"
            for index, row in frame.iterrows()
        )

    largest_rows = "\n".join(
        f"| {_money(row.price)} | {int(row.beds)} | {row.baths:g} | {row['size']:,.0f} | {row.lot_size:,.0f} | {int(row.zip_code)} |"
        for _, row in largest.iterrows()
    )
    correlation_rows = "\n".join(
        f"| {feature} | {correlations.loc[feature, TARGET]:.3f} |" for feature in strongest.index
    )

    return f"""# Seattle Dataset EDA Report

Generated from `{summary['data_path']}`. The script reports unusual observations but does not remove them.

## Dataset

- Rows: {summary['rows']:,}
- Columns: {len(summary['columns'])}
- ZIP codes: {summary['unique_zip_codes']}
- Rows where missing raw lot information is represented by zero: {summary['missing_lot_indicator_count']:,}
- Validation status: {'passed' if summary['validation']['valid'] else 'failed'}

The available source has no year built, renovation year, latitude, longitude, address, or property identifier. Analyses requiring those fields cannot be supported by this dataset.

## Target distribution

- Minimum: {_money(price['minimum'])}
- Median: {_money(price['median'])}
- Mean: {_money(price['mean'])}
- Maximum: {_money(price['maximum'])}
- Raw-price skew: {price['skew']:.3f}
- `log1p(price)` skew: {price['log1p_skew']:.3f}
- Conservative 3×IQR upper boundary: {_money(price['three_iqr_upper_boundary'])}
- Listings above that boundary: {price['three_iqr_outlier_count']}

The raw target is extremely right-skewed, largely because of a small luxury segment including a $25M observation. A log-target model is justified as an experiment, but it must be selected using cross-validation and evaluated back in dollars.

![Raw and log price distributions](figures/price_distribution.png)

## Feature relationships

![Feature relationships](figures/feature_relationships.png)

Spearman correlations are descriptive only; they are not model feature importance:

| Feature | Correlation with price |
|---|---:|
{correlation_rows}

![Correlation matrix](figures/correlations.png)

Living area has the clearest monotonic relationship with price. Beds and baths contain useful but overlapping size/capacity information. Lot size is highly skewed and weakly related to price across the whole market.

## Location

ZIP must be treated as categorical. Its numeric ordering has no defensible continuous meaning.

Highest median-price ZIPs:

| ZIP | Listings | Median | Mean |
|---|---:|---:|---:|
{zip_rows(top_zips)}

Lowest median-price ZIPs:

| ZIP | Listings | Median | Mean |
|---|---:|---:|---:|
{zip_rows(low_zips)}

![Median price by ZIP](figures/zip_median_price.png)

Any price-derived ZIP encoding must be fitted inside each training fold. Computing it once before cross-validation would leak fold targets.

## Largest target values

| Price | Beds | Baths | Size | Lot size | ZIP |
|---:|---:|---:|---:|---:|---:|
{largest_rows}

These rows are retained because the repository has no source field proving they are corrupt. Error analysis must report performance both overall and by price segment so that luxury observations do not hide typical-market behavior.

## Modeling implications

1. Compare raw and log-transformed targets using identical folds.
2. Use one-hot ZIP encoding as the safe baseline.
3. Keep all preprocessing inside the model pipeline.
4. Add only inference-reproducible ratios and missingness indicators.
5. Use MAE alongside RMSE because RMSE is dominated by the luxury tail.
6. Prefer repeated or nested CV for selection; the legacy test set has already been examined repeatedly.
"""


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(data)
    _save_price_distribution(data)
    _save_numeric_relationships(data)
    zip_summary = _save_zip_prices(data)
    correlations = _save_correlations(data)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(render_report(summary, zip_summary, correlations, data), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)} and {len(list(FIGURE_DIR.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
