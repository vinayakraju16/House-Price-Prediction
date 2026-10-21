# HavenValue frontend

## Cloudflare Workers (static assets)

This app is deployed as a Cloudflare Workers static-assets site, configured by
[`wrangler.toml`](wrangler.toml):

```toml
name = "house-price-prediction"
compatibility_date = "2026-09-19"

[assets]
directory = "./dist"
not_found_handling = "single-page-application"
```

Deploy with:

```powershell
npm run build
npx wrangler deploy
```

Do not deploy the `frontend` source directory itself: its `index.html` references uncompiled JSX
from `/src/index.jsx`, which browsers cannot execute and which Workers will serve as plain text
(`text/jsx`), producing a blank page. `wrangler deploy` must run after `npm run build`, uploading
only `frontend/dist`, whose scripts and styles are compiled under `/assets/`.

`not_found_handling = "single-page-application"` is required for React Router URLs such as
`/house-price` and `/about` to resolve — unlike Cloudflare Pages, Workers does not infer SPA
routing automatically; without this setting, a direct visit to those URLs returns a 404.

## The model runs in the browser

The estimate and "How it works" pages need no backend. The King County model (version 3.0.0)
is exported to JSON and evaluated client-side by [`src/model/predictor.js`](src/model/predictor.js),
a line-for-line port of `backend/api/services.py` and `backend/api/validation.py`: the same
feature engineering, median imputation, scaling, one-hot encoding and gradient-boosted trees,
plus the empirical price range, permutation-Shapley factors and nearest historical sales.
The JSON is loaded on demand, so the landing page stays small.

After retraining or registering a new model version, regenerate the exported files from the
repository root with the project venv (scikit-learn 1.4.2) and commit them:

```powershell
.\venv\Scripts\python.exe -m ml.training.export_browser_model --fixtures
```

This rewrites `src/model/king-county-<version>.json`, `src/model/comparables-<version>.json`
and `src/model/__fixtures__/parity-<version>.json`. The fixtures are scored by the real
scikit-learn pipeline, and `src/model/predictor.test.js` fails if the JavaScript port ever
disagrees with them. If the version changes, update the two import paths in
`src/model/index.js`.

The frontend makes no API calls. The Django API remains available for other clients and for
the Docker Compose stack.

React 18 client built with Vite and tested with Vitest + Testing Library.

## Local development

```powershell
npm ci
npm start
```

The development server listens on `http://localhost:3000`. No backend is needed: every page,
including estimates, runs entirely in the browser.

## Verification

```powershell
npm run lint
npm test
npm run build
```

- `npm test` runs the suite once for CI.
- `npm run test:watch` starts interactive watch mode.
- `npm run build` writes the production assets to `dist/`.

In Docker Compose, Nginx still proxies `/api/` to Django for API clients; the React app itself
does not use it.

## Browser end-to-end tests

The Playwright suite runs against the Docker Compose stack and checks that a valuation renders
without any prediction API request:

```powershell
npx playwright install chromium
cd ..
docker compose up --build --detach --wait --wait-timeout 180
cd frontend
npm run test:e2e
cd ..
docker compose down
```

Set `PLAYWRIGHT_BASE_URL` to test another running stack. Failure traces, screenshots, and videos are
written under `test-results/` and are excluded from Git and Docker build contexts.

With the same stack running, `npm run screenshots` regenerates the reviewed desktop and mobile
portfolio captures under `docs/screenshots/`. The capture fails on browser runtime errors.
