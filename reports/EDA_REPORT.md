# Exploratory Data Analysis — King County Seattle Cohort

## Scope

- Rows: **83,593**
- Columns: **21**
- Unique parcels: **68,409**
- Sale dates: **2015-01-02** through **2026-07-22**
- Median price: **$752,000**
- 1st–99th percentile: **$265,000–$2,925,000**

The raw target is strongly right-skewed (5.76); `log1p(price)` reduces skew to
0.20. Both raw and log targets were benchmarked. The log target won grouped CV for
the selected model family, so version 3.0.0 uses it and reverses predictions with `expm1`.

## Strongest univariate numeric relationships

- `size`: absolute Pearson correlation 0.670
- `grade`: absolute Pearson correlation 0.619
- `baths`: absolute Pearson correlation 0.496
- `finished_basement_sqft`: absolute Pearson correlation 0.366
- `beds`: absolute Pearson correlation 0.321
- `fireplaces`: absolute Pearson correlation 0.293

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
