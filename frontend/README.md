# HavenValue frontend

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
