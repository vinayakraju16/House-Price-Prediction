# King County Model Evaluation

The model was selected using five-fold `GroupKFold` cross-validation, grouping by parcel. Sales from 2025–2026 form a temporal holdout, and parcels appearing in that holdout were excluded from development data.

## Model comparison

| Model | CV MAE | CV RMSE | CV R² | Holdout MAE | Holdout RMSE | Holdout R² |
|---|---:|---:|---:|---:|---:|---:|
| Dummy median | $313,828 | $537,709 | -0.049 | $475,038 | $817,962 | -0.340 |
| Ridge rich | $121,200 | $246,054 | 0.778 | $247,544 | $486,672 | 0.526 |
| HistGradientBoosting five-field | $184,794 | $308,911 | 0.653 | $309,168 | $482,323 | 0.534 |
| HistGradientBoosting rich raw | $109,926 | $222,057 | 0.820 | $157,103 | $296,939 | 0.823 |
| HistGradientBoosting rich log | $104,563 | $214,331 | 0.832 | $152,425 | $293,251 | 0.828 |
| RandomForest rich log | $125,060 | $249,211 | 0.774 | $179,562 | $343,890 | 0.763 |

The bounded 12-trial randomized search selected a rich log-target HistGradientBoosting model with CV RMSE **$210,053**. Its untouched temporal-holdout metrics were MAE **$149,333**, RMSE **$289,585**, and R² **0.832**.

## Error by sale-price band

| Segment | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Below $500K | 275 | $82,293 | $112,739 | -3.511 |
| $500K-$1M | 4,396 | $79,477 | $108,684 | 0.339 |
| $1M-$2M | 3,290 | $169,874 | $222,760 | 0.241 |
| $2M+ | 676 | $530,899 | $864,841 | 0.483 |

## Uncertainty

The API interval uses the 5th and 95th percentiles of temporal-holdout residuals. Its measured holdout coverage is **90.0%** for a nominal 90% interval. This is an empirical population-level range, not a formal appraisal guarantee.

## Limitations

- Seattle coverage is approximated using King County ZIP codes 98101–98199.
- Public assessor sales and building attributes can lag corrections and renovations.
- Luxury homes remain harder to estimate and dominate RMSE.
- Current implementation has no precise latitude/longitude or neighborhood field.
