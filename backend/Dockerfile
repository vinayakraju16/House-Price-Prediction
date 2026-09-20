FROM python:3.10.20-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/backend

COPY requirements-backend.txt /app/requirements-backend.txt
RUN python -m pip install -r /app/requirements-backend.txt

COPY backend /app/backend
COPY data/Seattle/train_cleaned.csv /app/data/Seattle/train_cleaned.csv

RUN chmod +x /app/backend/entrypoint.sh && \
    mkdir -p /app/runtime /app/backend/staticfiles && \
    python manage.py collectstatic --noinput && \
    groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid appuser --no-create-home appuser && \
    chown -R appuser:appuser /app/runtime /app/backend/staticfiles

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/', timeout=3)"

ENTRYPOINT ["/app/backend/entrypoint.sh"]
