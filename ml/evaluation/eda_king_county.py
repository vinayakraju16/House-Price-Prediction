"""Generate reproducible EDA figures and findings for the active official dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "king-county" / "sales.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "EDA_REPORT.md"


def main() -> None:
    data = pd.read_csv(DATA_PATH, dtype={"zip_code": "string"})
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data["price"], bins=100, color="#31745a")
    axes[0].set(title="Sale-price distribution", xlabel="Price", ylabel="Sales")
    axes[0].set_xlim(0, data["price"].quantile(0.995))
    axes[1].hist(np.log1p(data["price"]), bins=80, color="#355f8a")
    axes[1].set(title="Log sale-price distribution", xlabel="log1p(price)", ylabel="Sales")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "price_distribution.png", dpi=160)
    plt.close(figure)

    sample = data.sample(n=min(30_000, len(data)), random_state=42)
    relationships = [
        ("beds", "Bedrooms"), ("baths", "Bathrooms"), ("size", "Living area"),
        ("lot_size", "Lot area"), ("year_built", "Year built"), ("grade", "Building grade"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (column, label) in zip(axes.flat, relationships, strict=True):
        axis.scatter(sample[column], sample["price"], s=3, alpha=0.12, color="#31745a")
        axis.set(xlabel=label, ylabel="Price")
        axis.set_ylim(0, data["price"].quantile(0.995))
        if column == "lot_size":
            axis.set_xlim(0, data[column].quantile(0.99))
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "feature_relationships.png", dpi=160)
    plt.close(figure)

    zip_summary = data.groupby("zip_code")["price"].agg(["median", "count"]).sort_values("median")
    figure, axis = plt.subplots(figsize=(13, 5))
    zip_summary["median"].plot.bar(ax=axis, color="#31745a")
    axis.set(title="Median sale price by ZIP", xlabel="ZIP code", ylabel="Median price")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "zip_median_price.png", dpi=160)
    plt.close(figure)

    correlation_columns = [
        "price", "beds", "baths", "size", "lot_size", "stories", "grade",
        "condition", "year_built", "finished_basement_sqft", "garage_sqft", "fireplaces",
    ]
    correlations = data[correlation_columns].corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlations)), correlations.columns, rotation=75, ha="right")
    axis.set_yticks(range(len(correlations)), correlations.columns)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "correlations.png", dpi=160)
    plt.close(figure)

    price_skew = float(data["price"].skew())
    log_skew = float(np.log1p(data["price"]).skew())
    top_correlations = correlations["price"].drop("price").abs().sort_values(ascending=False).head(6)
    correlation_text = "\n".join(
        f"- `{name}`: absolute Pearson correlation {value:.3f}"
        for name, value in top_correlations.items()
    )
    report = f"""# Exploratory Data Analysis — King County Seattle Cohort

## Scope

- Rows: **{len(data):,}**
- Columns: **{data.shape[1]}**
- Unique parcels: **{data['parcel_id'].nunique():,}**
- Sale dates: **{data['sale_date'].min()}** through **{data['sale_date'].max()}**
- Median price: **${data['price'].median():,.0f}**
- 1st–99th percentile: **${data['price'].quantile(.01):,.0f}–${data['price'].quantile(.99):,.0f}**

The raw target is strongly right-skewed ({price_skew:.2f}); `log1p(price)` reduces skew to
{log_skew:.2f}. Both raw and log targets were benchmarked. The log target won grouped CV for
the selected model family, so version 3.0.0 uses it and reverses predictions with `expm1`.

## Strongest univariate numeric relationships

{correlation_text}

ZIP has substantial price separation and is encoded categorically. Building grade, living area,
age, bathrooms, condition, basement, garage, and view information add signal beyond the original
five fields. Latitude/longitude and a reliable neighborhood field are not present in these extracts,
so the project does not invent them.

## Data-quality observations

- Invalid/non-arm's-length transfers are filtered using documented assessor codes before modeling.
- No price-quantile trimming is applied. Suspicious positive prices remain visible in the audit.
- `year_renovated` is missing when the assessor records zero/no renovation and is imputed inside CV.
- Repeated parcel sales are expected. CV groups by parcel and the temporal holdout's parcels are
  excluded from development data to avoid leakage.
- Extreme luxury properties dominate RMSE; segment metrics are reported in `MODEL_EVALUATION.md`.

Generated figures: `price_distribution.png`, `feature_relationships.png`,
`zip_median_price.png`, and `correlations.png` under `reports/figures/`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)} and {FIGURE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
