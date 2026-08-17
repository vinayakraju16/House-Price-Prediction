"""Lightweight database-backed inference monitoring summaries."""

import math
from collections import Counter
from datetime import timedelta

from django.utils import timezone

from .models import PredictionLog
from .services import FEATURE_NAMES, model_metadata


def _percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def monitoring_summary(days=7, recent_limit=20):
    """Aggregate operational metrics without returning stored property inputs."""
    days = max(1, min(int(days), 90))
    events = list(PredictionLog.objects.filter(created_at__gte=timezone.now() - timedelta(days=days)))
    successes = [event for event in events if event.status == "success"]
    errors = [event for event in events if event.status != "success"]
    latencies = [event.latency_ms for event in events]
    predictions = [event.prediction for event in successes if event.prediction is not None]

    feature_drift = _feature_drift_summary(successes)
    recent = [
        {
            "request_id": str(event.request_id),
            "timestamp": event.created_at.isoformat(),
            "endpoint": event.endpoint,
            "status": event.status,
            "model_version": event.model_version,
            "latency_ms": round(event.latency_ms, 2),
            "prediction": round(event.prediction, 2) if event.prediction is not None else None,
            "error_code": event.error_code or None,
        }
        for event in events[:recent_limit]
    ]
    return {
        "window_days": days,
        "generated_at": timezone.now().isoformat(),
        "requests": {
            "total": len(events),
            "successful": len(successes),
            "errors": len(errors),
            "error_rate": len(errors) / len(events) if events else 0.0,
        },
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
        },
        "predictions": {
            "count": len(predictions),
            "minimum": min(predictions) if predictions else None,
            "mean": sum(predictions) / len(predictions) if predictions else None,
            "maximum": max(predictions) if predictions else None,
        },
        "by_model_version": dict(Counter(event.model_version for event in events)),
        "errors_by_code": dict(Counter(event.error_code or "unknown" for event in errors)),
        "feature_monitoring": feature_drift,
        "recent": recent,
    }


def _feature_drift_summary(events):
    """Flag inputs outside training p05/p95; this is a heuristic, not a drift test."""
    try:
        ranges = model_metadata().get("typical_feature_ranges", {})
    except (OSError, RuntimeError, ValueError):
        ranges = {}
    summary = {}
    for feature in FEATURE_NAMES:
        values = [
            float(event.input_features[feature])
            for event in events
            if feature in event.input_features
        ]
        bounds = ranges.get(feature)
        outside = [] if not bounds else [
            value for value in values
            if value < float(bounds["p05"]) or value > float(bounds["p95"])
        ]
        summary[feature] = {
            "observed_count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "outside_training_p05_p95_rate": len(outside) / len(values) if values else None,
        }
    return summary
