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

Set `VITE_API_URL` to the public backend origin before building. The value is embedded in the
frontend bundle by Vite. No public backend origin is configured yet (see the root
`CLOUDFLARE_DEPLOYMENT_AUDIT.md`), so the deployed build currently falls back to
`http://localhost:8000`, which is unreachable from the deployed site — the prediction form will
show a connection error until `VITE_API_URL` is set to a real hosted backend and the site is
rebuilt and redeployed.

React 18 client built with Vite and tested with Vitest + Testing Library.

## Local development

```powershell
npm ci
npm start
```

The development server listens on `http://localhost:3000`. The client uses
`http://localhost:8000` as the default Django API origin. Override it before startup when needed:

```powershell
$env:VITE_API_URL = "https://api.example.com"
npm start
```

Vite injects this value at build time; changing it after `npm run build` does not rewrite an existing bundle.

## Verification

```powershell
npm run lint
npm test
npm run build
```

- `npm test` runs the suite once for CI.
- `npm run test:watch` starts interactive watch mode.
- `npm run build` writes the production assets to `dist/`.

Docker Compose builds with `VITE_API_URL=/api`, and Nginx proxies that same-origin path to Django.

## Browser end-to-end tests

The Playwright suite uses the real Nginx and Django containers, including the active model artifact:

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
portfolio captures under `docs/screenshots/`. The capture fails on browser runtime or prediction
request errors.
