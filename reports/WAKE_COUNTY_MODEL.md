# Wake County, NC Model

Development sales 2024-10-01 to 2026-03-31
(26,753 rows); temporal holdout 2026-04-01 to
2026-10-15 (9,206 rows). Holdout parcels are excluded
from development.

Models are chosen on a temporal validation inside development (temporal validation MAE (train before 2026-01-01, validate 2026-01-01 to 2026-04-01)).
Shuffled five-fold GroupKFold CV is shown for comparison only: it mixes months, so it
rewards features that merely encode *when* a sale happened, which then fail on later sales.

## Which data helps (feature ablation)

| Model | Shuffled CV MAE | Temporal validation MAE | Holdout MAE | Holdout RMSE | Holdout R² | Holdout median error | Within 10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Median price (baseline) | $204,311 | $205,167 | $227,188 | $407,878 | -0.072 | 31.1% | 17% |
| Property facts only | $61,132 | $64,464 | $71,462 | $162,470 | 0.830 | 7.9% | 60% |
| + Census ZIP income | $60,655 | $63,202 | $69,081 | $156,050 | 0.843 | 7.8% | 61% |
| + Zillow ZHVI | $60,413 | $63,027 | $70,328 | $157,288 | 0.841 | 7.9% | 60% |
| + ZHVI + inventory | $60,622 | $65,666 | $74,048 | $164,594 | 0.825 | 8.4% | 58% |
| + all market data | $60,195 | $64,460 | $73,087 | $165,953 | 0.823 | 8.3% | 58% |
| All market data, no calendar | $59,969 | $65,653 | $73,083 | $168,123 | 0.818 | 8.1% | 59% |
| ZHVI-relative: property + income | $59,843 | $62,208 | $69,973 | $159,886 | 0.835 | 7.7% | 61% |
| ZHVI-relative: + ZHVI momentum | $59,822 | $62,113 | $69,960 | $161,222 | 0.833 | 7.8% | 61% |
| ZHVI-relative: all market data | $60,231 | $64,711 | $70,700 | $162,149 | 0.831 | 7.8% | 60% |
| ZHVI-relative: all market + calendar | $60,236 | $64,129 | $71,064 | $163,239 | 0.828 | 7.9% | 60% |

Selected: **ZHVI-relative: + ZHVI momentum**, then tuned with
12 random trials on the same temporal validation
(untuned holdout MAE $69,960; small tuning gains on validation
do not always carry over to later months).

## Final model on the holdout

MAE **$70,361**, RMSE **$165,936**, R² **0.823**,
median error **7.8%**, within 10% of the sale price
**61%** of the time.

| Price band | Rows | MAE | Median error |
|---|---:|---:|---:|
| Below $300K | 1,260 | $54,430 | 10.8% |
| $300K-$500K | 4,008 | $34,675 | 6.6% |
| $500K-$800K | 2,543 | $60,782 | 7.6% |
| $800K+ | 1,395 | $204,744 | 11.1% |

| Sale type | Rows | MAE | Median error |
|---|---:|---:|---:|
| new | 1,927 | $49,259 | 5.7% |
| resale | 7,279 | $75,948 | 8.5% |

90% range: estimate × 0.82 to × 1.31
(holdout coverage 90%).

## Most important inputs (holdout permutation importance, MAE increase)

| Feature | MAE increase when shuffled |
|---|---:|
| size | $120,666 |
| zhvi | $66,104 |
| grade | $43,169 |
| median_household_income | $23,012 |
| lot_size | $14,069 |
| condition | $6,876 |
| property_type | $4,477 |
| owner_occupied_share | $4,374 |
| year_built | $3,579 |
| stories | $2,708 |
| zip_code | $1,352 |
| exterior_wall | $706 |

Runtime: Python 3.13.16, scikit-learn 1.9.1,
pandas 3.0.5.
