# HavenValue — Seattle home-price estimates, computed in your browser

[![CI](https://github.com/vinayakraju16/House-Price-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/vinayakraju16/House-Price-Prediction/actions/workflows/ci.yml)

**Live demo: [house-price-prediction.vinayakraju.dev](https://house-price-prediction.vinayakraju.dev)**

HavenValue estimates the sale price of a Seattle-area home from its size, location, quality and
age. The model is a gradient-boosted tree pipeline trained on **83,593 real King County Assessor
sales**. It runs **entirely in the visitor's browser**: the deployed site is a static Cloudflare
Workers app with no backend, and every estimate comes with:

- a measured 90% price range;
- a per-feature explanation of what drove the price;
- the five most similar historical sales;
- live "what-if" sliders that re-price the home as you drag.

![HavenValue estimate with empirical range, factors, and comparables](docs/screenshots/valuation-result-desktop.png)

## Highlights

- **Real data, privacy-safe.** The model joins the county's sales, residential-building and parcel
  extracts (2015 to July 2026). Owner names, addresses and recording numbers are dropped at read
  time ([`ml/data/king_county.py`](ml/data/king_county.py)).
- **Target leakage caught and fixed.** An earlier model reported R² 0.957, but two of its features
  were computed from the sale price itself. Measured the way the API actually runs it, it scored
  R² 0.499. It was retired and rebuilt
  ([`reports/PRODUCTION_MODEL_AUDIT.md`](reports/PRODUCTION_MODEL_AUDIT.md)).
- **Honest evaluation.** Models are chosen by 5-fold cross-validation grouped by parcel, so the same
  house never appears in both train and test. They are then scored on an untouched **2025–2026
  temporal holdout: R² 0.832, MAE $149k**. Errors on luxury homes are reported, not hidden
  ([`reports/MODEL_EVALUATION.md`](reports/MODEL_EVALUATION.md)).
- **Serverless inference with verified parity.** The fitted scikit-learn pipeline is exported to
  JSON and re-implemented in JavaScript. It matches scikit-learn to within 2×10⁻¹⁶ relative error
  on 21,786 homes, and a CI test fails if they ever diverge. Hosting cost: $0.
- **Explainable by default.** Each estimate includes 128-path permutation-Shapley contributions
  that add up exactly to the prediction, plus nearest comparable sales.
- **Production engineering.** The repo includes:
  - a checksummed model registry with rollback and versioned metadata;
  - a reproducible training CLI and optional MLflow tracking;
  - a Django API with request telemetry and drift monitoring;
  - Docker Compose, with an optional PostgreSQL setup;
  - GitHub Actions CI running Ruff, ML unit tests, Django tests on SQLite and PostgreSQL, ESLint,
    Vitest, and Playwright browser tests.

## Architecture

```mermaid
flowchart LR
    A[King County Assessor extracts] --> B[Privacy-safe joins and validation]
    B --> C[Grouped-CV benchmark and tuning]
    C --> D[Checksummed model registry]
    D --> E[export_browser_model.py]
    E --> F[Model JSON in the React app]
    F --> G[Browser: prediction, range, Shapley factors, comparables, what-if]
    D --> H[Django API: same model for other clients]
    H --> I[Telemetry and drift monitoring]
```

Training is an explicit offline step. The website never calls a server: Vite ships the model as
two lazily loaded JSON files (~2 MB, much smaller gzipped) that the browser downloads only when
someone asks for an estimate. The Django API serves the same registered artifact for other
clients and for the Docker Compose stack.

## Results

Model selection uses 5-fold `GroupKFold` by parcel. Parcels that appear in the 2025–2026 holdout
are excluded from development data.

| Model | CV MAE | CV R² | Holdout MAE | Holdout R² |
|---|---:|---:|---:|---:|
| Dummy median | $313,828 | −0.049 | $475,038 | −0.340 |
| Ridge (all features, log target) | $121,200 | 0.778 | $247,544 | 0.526 |
| Gradient boosting (5 basic fields) | $184,794 | 0.653 | $309,168 | 0.534 |
| Random forest (all features) | $125,060 | 0.774 | $179,562 | 0.763 |
| Gradient boosting (all features, log target) | $104,563 | 0.832 | $152,425 | 0.828 |
| **Same, after a 12-trial tuning search — deployed** | — | — | **$149,333** | **0.832** |

The tuning search improved CV RMSE from $214,331 to $210,053.

The 90% price range comes from holdout residual quantiles, and its measured coverage on the
holdout is 90.0%. Homes above $2M remain the hardest segment (holdout MAE $530,899). See
[`reports/MODEL_EVALUATION.md`](reports/MODEL_EVALUATION.md) for error by price band.

## How the in-browser model works

[`ml/training/export_browser_model.py`](ml/training/export_browser_model.py) reads the registered
pipeline and writes out its fitted parameters:
- median imputation values;
- standard-scaling means and scales;
- one-hot categories;
- every node of all 400 trees.

It also writes the bundled comparable sales and the fixed Shapley permutations.
[`frontend/src/model/predictor.js`](frontend/src/model/predictor.js) repeats each pipeline step
with the same floating-point operations. It also ports the API's validation rules and error
messages, so the site behaves exactly like the former `/predict/` endpoint.

Parity is checked three ways:
1. scikit-learn's own `predict`;
2. an independent Python evaluator that reads only the JSON;
3. the JavaScript, against
   [fixtures](frontend/src/model/__fixtures__/parity-3.0.0.json) scored by the backend's own
   explanation and comparable-search functions.

## Run it locally

The website needs only Node.js 22.13+:

```powershell
cd frontend
npm ci
npm run dev          # http://localhost:3000
```

For the full stack (Django API, Nginx, SQLite) with Docker:

```powershell
docker compose up --build --detach --wait --wait-timeout 180
# App: http://localhost:3000   API health: http://localhost:8000/health/   Readiness: http://localhost:8000/ready/
docker compose down
```

A validated PostgreSQL override is available with
`docker compose -f docker-compose.yml -f docker-compose.postgres.yml up --build`. Backups are
documented in [`reports/DEPLOYMENT.md`](reports/DEPLOYMENT.md).

To run the backend without Docker (Python 3.10, using the pinned scikit-learn 1.4.2):

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe backend\manage.py migrate
.\venv\Scripts\python.exe backend\manage.py runserver
```

## Retraining

Place the county's `Real Property Sales.zip`, `Residential Building.zip` and `Parcel.zip` under
`data/raw/king-county/` (Git ignores them), then run:

```powershell
.\venv\Scripts\python.exe -m ml.data.king_county                 # privacy-safe joined dataset
.\venv\Scripts\python.exe -m ml.training.benchmark_king_county   # grouped-CV comparison
.\venv\Scripts\python.exe -m ml.training.tune_king_county        # bounded randomized search
.\venv\Scripts\python.exe -m ml.training.train_king_county       # evaluate, fit, register
.\venv\Scripts\python.exe -m ml.training.export_browser_model --fixtures   # update the website
```

The registry ([`backend/mlmodels/model_registry.json`](backend/mlmodels/model_registry.json))
records SHA-256 hashes, and Django verifies them before loading a model. Set `MODEL_VERSION` to
pin or roll back the API. MLflow tracking is optional (`requirements-mlflow.txt`) and never
imported at inference time.

## Repository layout

| Path | Contents |
|---|---|
| `frontend/` | React 18 + Vite app, in-browser model (`src/model/`), Vitest and Playwright tests |
| `ml/` | Data preparation, feature pipelines, benchmarking, tuning, training, export |
| `backend/` | Django API, model registry (`mlmodels/`), monitoring, request telemetry |
| `reports/` | Evaluation, benchmark, leakage audit, operations and deployment notes |
| `docs/` | Screenshots and project/deployment audit write-ups |
| `data/Seattle/` | Small legacy dataset behind the `seattle-ridge-v1` rollback model and the original five-field pipeline |

## API (optional backend)

`POST /predict/` takes `beds`, `baths`, `size`, `lot_size` and `zip_code` (98101–98199). It also
accepts optional building details: `stories`, `grade`, `condition`, `year_built`,
`year_renovated`, `finished_basement_sqft`, `garage_sqft`, `fireplaces`, `has_view`,
`heat_system` and `property_type`. Omitted fields use the training defaults.

The response contains:
- `prediction`;
- the 90% `range`;
- input `confidence`;
- `top_factors` (Shapley contributions);
- five `comparables`.

Invalid input returns HTTP 400 with field-level `details`.

`GET /monitoring/?days=7` reports request counts, error rate, latency percentiles, the prediction
distribution, and an input-drift signal; send `X-Monitoring-Key` outside development
([`reports/OPERATIONS.md`](reports/OPERATIONS.md)). A separate, experimental Harris County (Texas)
appraisal-value model is served at `/harris-schema/` and `/predict-harris/`
([`ml/HARRIS_COUNTY.md`](ml/HARRIS_COUNTY.md)).

## Testing

```powershell
.\venv\Scripts\python.exe -m unittest discover -s ml\tests -v
.\venv\Scripts\ruff.exe check backend\api backend\djanoproject backend\model_runtime ml\data ml\features ml\training ml\evaluation ml\tests
cd backend; ..\venv\Scripts\python.exe manage.py test api; cd ..
cd frontend; npm run lint; npm test; npm run build
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs these checks plus the PostgreSQL
Django suite and Playwright end-to-end tests against the Docker Compose stack.

## Limitations and next steps

- **One range for every home.** The price range has the same width for every house. Quantile
  models would give a $3M home a wider range than a $500k condo.
- **No precise location.** The model knows only the ZIP code, not latitude/longitude or
  neighbourhood. Location features are the likeliest way to cut the large errors on luxury homes.
- **Model comparison.** A LightGBM/CatBoost comparison and a market-index adjustment for older
  sales are natural extensions.
- **Not an appraisal.** Comparables are historical assessor sales, not live listings or verified
  appraisal comparables.
