# Production Model Audit

Audit date: August 16, 2026

## Finding

The Phase 1 evaluation was not representative of production inference. Two engineered
features (`price_per_sqft` and `is_price_anomaly`) were calculated from the target sale
price. The API does not know that price, so it substituted constants. The held-out Phase 1
score therefore measured a different feature pipeline from the deployed request path.

Measured through the actual API-style transformation, the Phase 1 artifact achieved:

- R²: 0.4986
- RMSE: $430,850
- MAE: $272,501
- MAPE: 29.96%

The previously reported R² of 0.9567 and MAPE of 3.02% must not be used as production
claims.

## Corrective action

Production now uses the leakage-free Ridge baseline trained only from the five fields
available to the API. Its held-out metrics are R² 0.5492, RMSE $408,547, MAE $242,731,
and MAPE 27.15%. The feature-engineering script no longer creates target-derived model
inputs.

The model is registered as `seattle-ridge-v1`. Set `MODEL_VERSION` to a registered
version to select or roll back a deployment. Successful predictions are stored in the
`PredictionLog` table with the model version and base input features for monitoring.

## Subsequent P0 rebuild

Ridge remains available as a rollback version. The active experimental version is now
`2.0.0`, a tuned HistGradientBoosting pipeline selected with nested cross-validation.
Its nested CV MAE is $200,666 ± $38,069 and nested CV RMSE is $666,124 ± $483,074.
See `MODEL_BENCHMARK.md` and `MODEL_EVALUATION.md`; the invalid Phase 1 metrics remain
retired.
