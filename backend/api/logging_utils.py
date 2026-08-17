"""Small JSON logging formatter for machine-readable application logs."""

import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Serialize standard log records plus approved operational context."""

    context_fields = (
        "event", "request_id", "endpoint", "status", "model_version",
        "latency_ms", "error_code",
    )

    def format(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self.context_fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)
