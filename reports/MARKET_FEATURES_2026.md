# Market Data for 2026: What Was Added and What It Is Worth

October 2026. Measured on sales the models never saw (temporal holdouts), not on shuffled
cross-validation.

## The 2026 market these features describe

- **Rates rose again.** Freddie Mac's 30-year fixed went from 6.15% (2025-12-31) to 7.40%
  (2026-10-15), versus the low-6% range most forecasters expected for 2026.
- **Two-speed regional market.** Economists describe Sun Belt and Western metros as softer
  and the Midwest/Northeast as tighter. Wake County fits the soft side: its ZIPs' Zillow
  index averaged −0.8% year over year at sale time (−1.3% in September 2026, weighted by
  sales); active listings averaged +44% year over year during the sales period (growth
  has slowed to +8% now); and 24% of listings carried a price cut in September 2026.
- **New construction behaves differently.** About a third of Wake qualified sales (32%)
  were homes sold within a year of being built, falling from 42% in late 2024 to 24% in
  2026. The model prices them more accurately (5.7% median error vs 8.5% for resales).

## Data added (all free, national, ZIP level)

| Source | Features | Refresh |
|---|---|---|
| Zillow ZHVI (ZIP, mid-tier, SA) | `zhvi`, `zhvi_yoy` | Monthly |
| Realtor.com inventory (ZIP) | `days_on_market`, `price_reduced_share`, `inventory_yoy`, `pending_ratio` | Monthly |
| Freddie Mac PMMS | `mortgage_rate_30y`, `mortgage_rate_change_1y` | Weekly → monthly |
| Census ACS 2020–2024 (ZCTA) | `median_household_income`, `owner_occupied_share` | Yearly |
| Derived from property facts | `is_new_construction`, `property_age`, `years_since_renovation`, `living_lot_ratio` | — |

Every sale only sees market data from the month before it closed.

## Results

### Wake County, NC (new model, `ml/training/train_wake_county.py`)

Holdout 2026-04-01 to 2026-10-15, 9,206 sales. Full table in `WAKE_COUNTY_MODEL.md`.

| Model | Holdout MAE | Median error | Within 10% |
|---|---:|---:|---:|
| Property facts only | $71,462 | 7.9% | 60% |
| + Census ZIP income | $69,081 | 7.8% | 61% |
| + Zillow ZHVI | $70,328 | 7.9% | 60% |
| + ZHVI + inventory | $74,048 | 8.4% | 58% |
| + all market data | $73,087 | 8.3% | 58% |
| **ZHVI-relative target + income + ZHVI momentum (selected, tuned)** | **$70,361** | **7.8%** | **61%** |

### King County / Seattle (deployed 3.0.0, evaluation only, `ml/training/evaluate_king_county_market.py`)

Holdout 2025-01-01 to 2026-07-22, 8,637 sales, same tuned hyperparameters as 3.0.0.

| Model | Holdout MAE | RMSE | R² | Median error, 2025 | Median error, 2026 |
|---|---:|---:|---:|---:|---:|
| Deployed 3.0.0 recipe (refit) | $149,122 | $288,779 | 0.833 | 8.7% | 10.4% |
| + Zillow ZHVI | $149,256 | $291,398 | 0.830 | 8.8% | 10.2% |
| + all market data | $149,820 | $289,950 | 0.832 | 9.0% | 10.0% |
| ZHVI-relative target, no calendar | $151,232 | $282,916 | 0.840 | 8.9% | 10.3% |

## What this means

1. **Census ZIP income is the clearest win** (Wake: −3% MAE). It is stable and costs
   nothing to keep current.
2. **Inventory, days on market and mortgage rates hurt as model inputs** in both markets.
   They change month to month for the whole market, so the model learns "when" a sale
   happened instead of "what" was sold, and that does not carry to later months. Shuffled
   cross-validation hid this (it showed them helping); a time-ordered validation exposed it.
   Keep them for display ("market is cooling: 27% of listings cut prices") rather than as
   inputs, and re-test once there are several more years of history.
3. **The ZHVI-relative target is the right long-run design.** It predicts price relative to
   the ZIP's current Zillow value, so when a ZIP cools or heats up the estimate moves with
   it at the next monthly ZHVI refresh, without retraining. Accuracy is about the same
   today; the gain is staying current.
4. **For Seattle, the deployed model is ageing faster than features can fix:** its median
   error rose from 8.7% (2025) to 10.4% (2026). Retraining 3.0.0 with 2025 sales (holding
   out 2026) is the biggest available improvement there.
5. Wake County has no bedroom count; Seattle has no new-construction effect worth
   measuring yet. Per-market models are the right structure.

## Not added, and why

- **Flood zone / insurance cost:** highly relevant in 2026 (insurance is a growing share of
  ownership cost in NC), but needs parcel coordinates and an address input in the app.
- **School ratings, walkability:** paid or location-level data; same address requirement.
- **Assessed value:** strong predictor, but users would have to look it up.
- **Zillow rent index (ZORI):** expected to add little over ZHVI and income.
