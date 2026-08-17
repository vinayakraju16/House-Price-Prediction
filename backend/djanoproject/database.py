"""Validated database configuration for local SQLite and production PostgreSQL."""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured


def _integer(environment, name, default, minimum=0):
    raw_value = environment.get(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ImproperlyConfigured(f"{name} must be an integer.") from exc
    if value < minimum:
        raise ImproperlyConfigured(f"{name} must be at least {minimum}.")
    return value


def database_settings(environment=None, base_dir=None):
    """Build Django's default database settings without exposing credentials."""
    environment = os.environ if environment is None else environment
    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
    engine = environment.get("DJANGO_DB_ENGINE", "sqlite").strip().lower()

    if engine in {"sqlite", "sqlite3"}:
        configured_path = environment.get("DJANGO_DB_PATH", "").strip()
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": configured_path or base_dir / "db.sqlite3",
        }

    if engine not in {"postgres", "postgresql"}:
        raise ImproperlyConfigured(
            "DJANGO_DB_ENGINE must be one of: sqlite, sqlite3, postgres, postgresql."
        )

    required = {
        "POSTGRES_DB": environment.get("POSTGRES_DB", "").strip(),
        "POSTGRES_USER": environment.get("POSTGRES_USER", "").strip(),
        "POSTGRES_PASSWORD": environment.get("POSTGRES_PASSWORD", ""),
        "POSTGRES_HOST": environment.get("POSTGRES_HOST", "").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ImproperlyConfigured(
            f"PostgreSQL configuration is missing required variables: {', '.join(missing)}."
        )

    options = {}
    ssl_mode = environment.get("POSTGRES_SSLMODE", "").strip()
    if ssl_mode:
        options["sslmode"] = ssl_mode

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": required["POSTGRES_DB"],
        "USER": required["POSTGRES_USER"],
        "PASSWORD": required["POSTGRES_PASSWORD"],
        "HOST": required["POSTGRES_HOST"],
        "PORT": _integer(environment, "POSTGRES_PORT", 5432, minimum=1),
        "CONN_MAX_AGE": _integer(environment, "POSTGRES_CONN_MAX_AGE", 60),
        "CONN_HEALTH_CHECKS": True,
        "OPTIONS": options,
    }
