# Cloudflare Deployment & Repository Audit

Audit performed: September 19, 2026

## Summary

The live Cloudflare Workers deployment (`house-price-prediction.vinayakraju01.workers.dev`)
was serving a blank page. The cause, fix, and verification are below, followed by a broader
repository audit (backend, ML, CI, Docker) that verified the claims in the earlier
`PROJECT_AUDIT.md` and found a handful of additional issues, most of which are fixed in this
change.

## Cloudflare deployment error: root cause and fix

**Symptom:** `https://house-price-prediction.vinayakraju01.workers.dev/` rendered a blank
page with `<div id="root"></div>` and nothing inside it.

**Root cause:** the Worker had been deployed from `frontend/` directly (source), not from
`frontend/dist/` (the Vite production build). The source `index.html` loads
`<script type="module" src="/src/index.jsx">` — uncompiled JSX that browsers cannot execute.
Cloudflare served that file as `Content-Type: text/jsx`, so the module script failed to load
and React never mounted. There was no `wrangler.toml` committed anywhere in the repository, so
each deploy depended on whichever directory was passed manually to `wrangler deploy`.

**Fix:**
- Added [`frontend/wrangler.toml`](frontend/wrangler.toml), pinning the deploy to the build
  output:
  ```toml
  name = "house-price-prediction"
  compatibility_date = "2026-09-19"

  [assets]
  directory = "./dist"
  not_found_handling = "single-page-application"
  ```
  `not_found_handling = "single-page-application"` is also required for direct navigation to
  React Router routes (`/about`, `/house-price`) — Cloudflare Workers, unlike Cloudflare Pages,
  does not infer SPA fallback automatically; without it, those URLs 404 on refresh or direct
  visit.
- Rebuilt (`npm run build`) and redeployed (`npx wrangler deploy`).
- Corrected `frontend/README.md`, which had documented this as a Cloudflare **Pages** project
  with automatic SPA detection — the actual deployment is a Workers static-assets site, which
  needed the explicit config above.

**Verification (live, post-fix):**

| Request | Before | After |
|---|---|---|
| `GET /` | 200, raw `index.html` referencing `/src/index.jsx` | 200, compiled `<script>`/`<link>` under `/assets/` |
| `GET /src/index.jsx` | 200, `Content-Type: text/jsx` (unusable in-browser) | not applicable to the built app |
| `GET /about` | — | 200 (SPA fallback) |
| `GET /house-price` | — | 200 (SPA fallback) |
| `GET /assets/index-*.js` | — | 200, `Content-Type: text/javascript` |

### Known follow-up (not a Cloudflare error, a product gap)

No backend is publicly hosted yet — `PROJECT_AUDIT.md`'s roadmap already flagged "add a
configured deployment URL after a hosting target is selected" as outstanding. The deployed
frontend therefore falls back to `VITE_API_URL`'s default, `http://localhost:8000`, which is
unreachable from a visitor's browser: the property estimate form will show a connection error
until a backend is hosted somewhere public, `VITE_API_URL` is set to that origin, and the
frontend is rebuilt and redeployed. This is a deployment-topology decision, not something to
default without input.

## Repository audit findings

`PROJECT_AUDIT.md` (Aug 16) is stale in places: its "Current measured evidence" table and
"Seattle data findings" describe an earlier, superseded phase of the project (the 2,016-row
raw Seattle dataset and a Ridge baseline), while its executive summary correctly describes the
active King County v3.0.0 model. Verified against `backend/mlmodels/versions/3.0.0/metadata.json`,
the executive summary is accurate; the two sections above it should be read as historical only.

### Fixed in this change

| Issue | File(s) | Fix |
|---|---|---|
| 12 compiled `.pyc` files were tracked in git despite `__pycache__/` being gitignored | `backend/**/__pycache__/*.pyc` | Untracked with `git rm --cached` |
| `matplotlib` (EDA-only, never imported by `backend/`) was installed into the production Django Docker image | `requirements.txt`, `backend/Dockerfile`, `.github/workflows/ci.yml` | Split into `requirements-backend.txt` (no matplotlib); `requirements.txt` now includes it via `-r requirements-backend.txt` plus `matplotlib`. Verified: fresh venv install of `requirements-backend.txt` + all 26 backend tests pass with matplotlib absent. |
| `.env.example` had no example CORS/CSRF origin for a Cloudflare-hosted frontend, and didn't document `WEB_CONCURRENCY`/`GUNICORN_TIMEOUT` (read by `backend/entrypoint.sh`) | `.env.example` | Added a Cloudflare Workers origin example and the two gunicorn tuning vars |
| `docker-compose.yml`'s placeholder `DJANGO_SECRET_KEY`/`MONITORING_API_KEY` fallbacks pass Django's "non-development secret" check, so a copy-pasted compose file could silently run a real deployment with a publicly-known secret | `docker-compose.yml` | Added an explicit comment warning these fallbacks are local-demo-only and must be overridden before any shared/public deployment |
| `engineer_features_for_inference` in `services.py` duplicates the pipeline's real feature engineering but is dead code on the live `/predict/` path (only a leakage-guard test calls it), which could mislead a future reader into thinking leakage protection lives there | `backend/api/services.py` | Left in place (removing it would need to touch its test) but re-documented as legacy/test-only, pointing to where the real feature engineering lives |

### Documented, not changed (see rationale)

These are real hardening recommendations, but the backend has no public deployment today, and
each one is either tied to an explicit existing test or an explicitly documented local
quick-start workflow. Changing them now risked breaking a documented flow without a live
deployment to validate against — recorded here as a pre-deployment checklist instead:

- **`DJANGO_DEBUG` defaults to `true`** (`backend/djanoproject/settings.py:17`) when
  `DJANGO_DEBUG` is unset. Docker Compose already hardcodes `DJANGO_DEBUG: "false"` explicitly,
  so this only matters for a bare (non-Docker) deployment. The documented local quick start
  (`manage.py runserver` with no env vars) and CI both rely on the current default; flipping it
  would raise `ImproperlyConfigured` for anyone following the README's quick start as-is, since
  it also doesn't set `DJANGO_SECRET_KEY`. **Before deploying the Django app anywhere public
  without Docker, explicitly set `DJANGO_DEBUG=false` and a real `DJANGO_SECRET_KEY`.**
- **`/monitoring/` is unauthenticated when `MONITORING_API_KEY` is empty**
  (`backend/api/views.py:260`), which is its default in `settings.py`. This is intentional,
  tested behavior for local development (`backend/api/tests.py:276`), not a bug — but it means
  forgetting to set the key in a real deployment silently exposes request volumes, error rates,
  and prediction statistics. **Always set `MONITORING_API_KEY` before exposing the backend
  beyond localhost** (now called out directly in `.env.example`).
- **`scikit-learn==1.4.2`** (`requirements.txt`/`requirements-backend.txt`) is a 2024 release,
  notably older than every other pin in the file. It intentionally matches the version the
  active model artifact was trained and pickled with — bumping it without retraining risks a
  joblib/pickle deserialization mismatch. Any future bump should happen alongside a documented
  retraining, not as a standalone dependency update.
- The backend Docker base image (`python:3.10.20-slim-trixie`) currently reports known OS-level
  CVEs. Rebuilding on a newer base wasn't validated in this pass (no way to exercise the full
  image build + runtime here) and is deployment-sensitive; do this as its own change with a
  full container smoke test.

### Verified clean (no action needed)

- No secrets or API keys are committed in tracked files — only clearly-labeled CI-only/local-only
  placeholders (verified by reading `settings.py`, `docker-compose*.yml`, `.env.example`, and a
  repo-wide search).
- Model artifact loading validates SHA-256 checksums against `model_registry.json` before
  deserializing, and rejects path traversal in registry entries
  (`backend/api/services.py:60-89`).
- Target-derived leakage features (`price_per_sqft`, `is_price_anomaly`) are structurally
  excluded from the trained pipeline, not just filtered at request time — confirmed against
  `backend/mlmodels/versions/3.0.0/metadata.json` and `ml/tests/test_artifact.py`.
- All `os.environ`/`os.getenv` keys read by `backend/` have a corresponding entry in
  `.env.example`.
- `.github/workflows/ci.yml` runs lint, unit tests, a Postgres integration job, and a full
  Docker Compose + Playwright end-to-end job on every push/PR; all jobs are well-formed.
- `ml/` path handling resolves paths relative to the repository root rather than hardcoding
  absolute paths, and every referenced data file exists on disk.

## Verification performed in this change

- `cd frontend && npm run build` — succeeds, produces compiled `dist/`.
- `npx wrangler deploy` — succeeds; live site re-verified with `curl` (table above).
- Fresh scratch venv: `pip install -r requirements-backend.txt` then
  `python manage.py test api` — 26/26 backend tests pass without matplotlib installed.
- `docker compose config` — validates without error after the `docker-compose.yml` comment
  addition.
