# Deployment Readiness

## Validated local stack

The repository contains separate backend and frontend images orchestrated by
`docker-compose.yml`:

- Django runs under Gunicorn as a non-root user on port 8000.
- Nginx serves the compiled React bundle on port 80 and proxies `/api/` to Django.
- `/health/` checks process liveness without loading dependencies.
- `/ready/` checks the selected database plus the registered model and metadata versions.
- migrations run before Gunicorn starts;
- the default SQLite database lives in the `backend_runtime` named volume.

The validated local command is:

```powershell
docker compose up --build
```

During verification, both containers became healthy and a prediction sent through
`http://localhost:3000/api/predict/` returned model version 2.2.0, five explanation factors,
and five historical comparables.

The browser-level verification uses that same stack:

```powershell
cd frontend
npx playwright install chromium
cd ..
docker compose up --build --detach --wait --wait-timeout 180
cd frontend
npm run test:e2e
cd ..
docker compose down
```

It validates landing-page navigation, direct SPA routing, a real prediction submission, the
empirical range, local factors, and comparable-property rendering.

## PostgreSQL stack

SQLite remains the default for a zero-setup demonstration. The additive PostgreSQL override uses
the pinned official `postgres:17.10-alpine3.23` image, a health-gated backend dependency, and a
separate `postgres_data` volume. The database port is not published to the host.

```powershell
$env:POSTGRES_PASSWORD = "replace-with-a-long-random-password"
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up --build --detach --wait --wait-timeout 240
docker compose -f docker-compose.yml -f docker-compose.postgres.yml ps
```

Django selects PostgreSQL only when `DJANGO_DB_ENGINE=postgresql`. Required settings are
`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_HOST`; port defaults to 5432.
`POSTGRES_CONN_MAX_AGE` controls persistent connections and connection health checks are enabled.
The local override explicitly disables PostgreSQL TLS inside the private Compose network. For a
managed database, set `POSTGRES_SSLMODE=require` or `verify-full` as required by the provider.

Stop the stack without deleting its database volume:

```powershell
docker compose -f docker-compose.yml -f docker-compose.postgres.yml down
```

Do not add `--volumes` unless permanent deletion of the PostgreSQL data is intended.

## PostgreSQL backup and restore

The local PostgreSQL service mounts `backups/postgres/` at `/backups`. Backup files are ignored by
Git and the Docker build context. Create an online custom-format backup with:

```powershell
.\scripts\postgres\backup.ps1
# Or choose an auditable filename:
.\scripts\postgres\backup.ps1 -OutputName before-release-20260816.dump
```

Restore is intentionally guarded because it cleans and replaces objects in the configured
database. Stop application writes, keep the database service running, and explicitly confirm:

```powershell
docker compose -f docker-compose.yml -f docker-compose.postgres.yml stop frontend backend
.\scripts\postgres\restore.ps1 -BackupName before-release-20260816.dump -ConfirmDatabaseReset
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up --detach --wait
```

The scripts reject directory traversal and operate only on simple `.dump` filenames directly in
`backups/postgres/`. A validated restore recovered a deliberately deleted database marker and the
application passed readiness and prediction checks afterward. For production, also encrypt and
copy backups off-host, apply retention rules, monitor backup jobs, and schedule recurring restore
drills. Managed database snapshots should complement—not replace—tested logical backups.

## Production configuration

Do not deploy with the Compose placeholder values. At minimum configure:

- a long random `DJANGO_SECRET_KEY`;
- `DJANGO_DEBUG=false` and exact `DJANGO_ALLOWED_HOSTS`;
- exact CORS and CSRF trusted origins;
- HTTPS termination, secure cookies, SSL redirect, proxy-proto trust, and HSTS;
- a strong `MONITORING_API_KEY`;
- a unique PostgreSQL password kept in the deployment secret manager;
- encrypted, off-host database backups and recurring restore tests.

Run the deployment checks with the intended production environment:

```powershell
cd backend
..\venv\Scripts\python.exe manage.py check --deploy --fail-level WARNING
```

HSTS should only be enabled after HTTPS is confirmed because browsers cache it. The application
trusts `X-Forwarded-Proto` only when `DJANGO_TRUST_PROXY_PROTO=true` is explicitly configured.

## Scaling constraints

SQLite is suitable for this portfolio demonstration and a single backend instance. Use the
validated PostgreSQL path before multiple Gunicorn containers or write-heavy traffic. The included
Compose database demonstrates the contract; a managed database with provider backups, TLS, patching,
and availability controls is preferred for a public deployment. LocMemCache is per-process, so
shared caching also requires an external cache if replicas are introduced.

The model artifact is bundled with the backend image and verified by SHA-256 before loading.
Deploy a new immutable image for model updates, or pin a registered rollback version through
`MODEL_VERSION` and restart every worker.

## CI/CD boundary

GitHub Actions validates code, tests against both SQLite and a PostgreSQL service, deployment
settings, frontend assets, container builds, and the Playwright workflow against a healthy Compose
stack. Failure diagnostics include container logs. No deployment job is included because no hosting
target or credential boundary has been chosen. Secrets must be configured in the eventual
deployment platform, never in the workflow or image.
