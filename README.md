# House Price Prediction

A React interface backed by a Django API that estimates a house price from 14 property attributes. The deployed API uses the best held-out model, Gradient Boosting.

## Project layout

- `frontend/` — React user interface.
- `backend/` — Django API and the deployed model bundle in `backend/mlmodels/`.
- `ml/train.py` — repeatable model training script.
- `ml/notebooks/` — exploratory notebooks retained for reference.
- `ml/archive/` — legacy feature-selection outputs and superseded model artifacts; these are not used by the application.

Only `backend/mlmodels/trained_models.pkl` is used by the application. The root-level selected-feature dataset, `final_14_selected_features.csv`, is the input used to retrain it.

Retrain that artifact reproducibly with `python ml/train.py`. This exports the model bundle and its held-out metrics to `backend/mlmodels/metadata.json`.

## Repository hygiene

`.gitignore` prevents environments, installed packages, logs, local databases, and frontend build output from being added again. Existing tracked generated files must be removed from Git's index once you have reviewed the current working-tree deletions; they are intentionally not removed automatically by this refactor.

## Run locally

Create and activate a virtual environment, then install the backend dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd backend
python manage.py runserver
```

In a second terminal:

```powershell
cd frontend
npm install
npm start
```

The frontend defaults to `http://localhost:8000`. Set `REACT_APP_API_URL` when using a different API origin.

## API

`POST /predict/` accepts a JSON object containing the 14 form fields and returns:

```json
{"prediction": 234567.89}
```

The response also includes a transparent error-based range and an input-based confidence label. Both are experimental guidance only, not a formal valuation or appraisal.

Invalid or missing values return a clear `400` response. Run backend tests with `python manage.py test` from `backend/`.

## Configuration

Set these environment variables for deployment: `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=false`, `DJANGO_ALLOWED_HOSTS`, and `CORS_ALLOWED_ORIGINS`.

## Optional live market context

The estimate form can also accept a U.S. address. The backend uses the Census Geocoder to identify its county and caches the lookup. Set `CENSUS_API_KEY` to show the county's ACS median home value and `FRED_API_KEY` to show the latest 30-year mortgage rate. These external values provide context only; they do not alter the trained property-model estimate.

MLS listings and sold comparables are deliberately not scraped. Connect a licensed MLS/RESO or commercial data provider before displaying them.

## Harris County / Houston data path

The next supported market is Harris County, Texas. The repository includes an HCAD public-data importer; see `data/README.md` and `ml/HARRIS_COUNTY.md`. HCAD files support an appraisal-value prototype and must not be presented as verified sale-price data.
