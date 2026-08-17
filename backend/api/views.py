import json
import logging
import secrets
import uuid
from time import perf_counter

from django.conf import settings
from django.db import connection
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .models import PredictionLog
from .monitoring import monitoring_summary
from .services import (
    estimate_details,
    harris_estimate,
    harris_metadata,
    load_models,
    market_context,
    model_metadata,
)
from .validation import PredictionInputError

logger = logging.getLogger(__name__)


def health(request):
    """Process liveness probe; deliberately avoids database and model work."""
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)
    return JsonResponse({"status": "ok", "service": "house-price-api"})


def readiness(request):
    """Dependency readiness probe for the database and active model artifact."""
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        bundle = load_models()
        metadata = model_metadata()
        if bundle["model_version"] != metadata.get("model_version"):
            raise RuntimeError("Model bundle and metadata versions differ.")
    except Exception:
        logger.exception("Readiness check failed", extra={"event": "readiness_failed"})
        return JsonResponse({"status": "unavailable"}, status=503)
    return JsonResponse({"status": "ready", "model_version": bundle["model_version"]})


def _request_id(request):
    candidate = request.headers.get("X-Request-ID", "")
    try:
        return str(uuid.UUID(candidate))
    except (ValueError, AttributeError):
        return str(uuid.uuid4())


def _response(payload, status, request_id):
    result = JsonResponse({**payload, "request_id": request_id}, status=status)
    result["X-Request-ID"] = request_id
    return result


def _validation_response(error, request_id):
    return _response(
        {
            "error": "validation_error",
            "message": str(error),
            "details": error.details,
        },
        400,
        request_id,
    )


def _malformed_json_response(request_id):
    message = "Request body must contain valid JSON."
    return _response(
        {"error": "malformed_json", "message": message, "details": {"body": message}},
        400,
        request_id,
    )


def _current_model_version():
    try:
        return model_metadata().get("model_version", "unknown")
    except Exception:
        return "unknown"


def _record_inference(*, request_id, endpoint, status, started_at, model_version,
                      error_code="", input_features=None, estimate=None):
    latency_ms = (perf_counter() - started_at) * 1000
    fields = {
        "request_id": request_id,
        "endpoint": endpoint,
        "status": status,
        "model_version": model_version,
        "latency_ms": round(latency_ms, 2),
        "error_code": error_code or None,
    }
    log_method = logger.info if status == "success" else logger.warning
    log_method("Inference request completed", extra={"event": "inference_completed", **fields})
    try:
        PredictionLog.objects.create(
            request_id=request_id,
            endpoint=endpoint,
            status=status,
            error_code=error_code,
            latency_ms=latency_ms,
            input_features=input_features or {},
            prediction=estimate["prediction"] if estimate else None,
            range_low=estimate["range"]["low"] if estimate else None,
            range_high=estimate["range"]["high"] if estimate else None,
            confidence_level=estimate["confidence"]["level"] if estimate else "",
            model_version=model_version,
        )
    except Exception:
        logger.exception("Inference event could not be stored", extra={"event": "monitoring_write_failed", **fields})


def model_info(request):
    """Return training metrics and global feature importance for the UI."""
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)
    metadata = model_metadata()
    return JsonResponse({
        "deployed_model": metadata["deployed_model"],
        "deployed_model_type": metadata.get("deployed_model_type", metadata["deployed_model"]),
        "dataset_rows": metadata["dataset_rows"],
        "training_rows": metadata["training_rows"],
        "test_rows": metadata["test_rows"],
        "model_version": metadata.get("model_version", "legacy"),
        "status": metadata.get("status", "unknown"),
        "training_date": metadata.get("training_date"),
        "feature_count": metadata.get("feature_count", len(metadata.get("features", []))),
        "metrics": metadata.get("metrics") or metadata.get("ridge_results") or metadata.get("gradient_boost_results", {}),
        "nested_cv": metadata.get("nested_cv", {}),
        "cross_validation": metadata.get("cross_validation", {}),
        "prediction_interval": metadata.get("prediction_interval", {}),
        "target_summary": metadata["target_summary"],
        "feature_importance": metadata["feature_importance"][:6],
    })


def harris_schema(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)
    metadata = harris_metadata()
    return JsonResponse({
        "market": metadata["market"], "appraisal_year": metadata["appraisal_year"],
        "features": metadata["features"], "categorical_features": metadata["categorical_features"],
        "numeric_features": metadata["numeric_features"], "example_input": metadata["example_input"],
        "metrics": metadata["metrics"], "disclaimer": metadata["target_disclaimer"],
    })


@require_POST
@csrf_exempt
def predict_harris(request):
    started_at = perf_counter()
    request_id = _request_id(request)
    try:
        payload = json.loads(request.body)
        estimate = harris_estimate(payload)
    except (TypeError, json.JSONDecodeError):
        _record_inference(request_id=request_id, endpoint="predict_harris", status="error",
                          started_at=started_at, model_version="harris-2026", error_code="malformed_json")
        return _malformed_json_response(request_id)
    except PredictionInputError as error:
        _record_inference(request_id=request_id, endpoint="predict_harris", status="error",
                          started_at=started_at, model_version="harris-2026", error_code="validation_error")
        return _validation_response(error, request_id)
    except Exception:
        logger.exception("Harris County prediction failed", extra={"event": "inference_failed", "request_id": request_id})
        _record_inference(request_id=request_id, endpoint="predict_harris", status="error",
                          started_at=started_at, model_version="harris-2026", error_code="model_unavailable")
        return _response({"error": "Harris County prediction service is unavailable."}, 503, request_id)
    _record_inference(request_id=request_id, endpoint="predict_harris", status="success",
                      started_at=started_at, model_version="harris-2026", estimate=estimate)
    return _response({
        "prediction": round(estimate["prediction"], 2),
        "range": {key: round(value, 2) if isinstance(value, (int, float)) else value for key, value in estimate["range"].items()},
        "confidence": estimate["confidence"], "disclaimer": estimate["disclaimer"],
    }, 200, request_id)


@require_POST
@csrf_exempt
def predict(request):
    """Return a house-price estimate for a validated JSON request."""
    started_at = perf_counter()
    request_id = _request_id(request)
    try:
        payload = json.loads(request.body)
    except (TypeError, json.JSONDecodeError):
        _record_inference(request_id=request_id, endpoint="predict", status="error",
                          started_at=started_at, model_version=_current_model_version(), error_code="malformed_json")
        return _malformed_json_response(request_id)

    try:
        estimate = estimate_details(payload)
    except PredictionInputError as error:
        _record_inference(request_id=request_id, endpoint="predict", status="error",
                          started_at=started_at, model_version=_current_model_version(), error_code="validation_error")
        return _validation_response(error, request_id)
    except Exception:
        logger.exception("House-price prediction failed", extra={"event": "inference_failed", "request_id": request_id})
        _record_inference(request_id=request_id, endpoint="predict", status="error",
                          started_at=started_at, model_version=_current_model_version(), error_code="model_unavailable")
        return _response({"error": "Prediction service is unavailable."}, 503, request_id)

    result = {
        "prediction": round(estimate["prediction"], 2),
        "range": {key: round(value, 2) if isinstance(value, (int, float)) else value for key, value in estimate["range"].items()},
        "confidence": estimate["confidence"],
        "model_version": estimate.get("model_version", "legacy"),
        "model_factors": estimate.get("model_factors", []),
        "top_factors": estimate.get("model_factors", []),
        "explanation": estimate.get("explanation", {"method": "unavailable", "top_factors": []}),
        "comparables": estimate.get("comparables", []),
        "disclaimer": "Experimental estimate based on King County Assessor sale records; not a formal appraisal.",
    }

    pincode = (
        payload.get("pincode")
        or payload.get("zipcode")
        or payload.get("postal_code")
        or payload.get("postalCode")
    )
    if payload.get("address"):
        result["market"] = market_context(payload["address"])
    elif pincode:
        result["market"] = market_context(str(pincode).strip())
    _record_inference(
        request_id=request_id,
        endpoint="predict",
        status="success",
        started_at=started_at,
        model_version=estimate.get("model_version", "legacy"),
        input_features={
            name: payload[name]
            for name in model_metadata().get("features", ())
            if name in payload
        },
        estimate=estimate,
    )
    return _response(result, 200, request_id)


@require_GET
def monitoring(request):
    """Return privacy-safe operational aggregates for an authorized operator."""
    configured_key = settings.MONITORING_API_KEY
    supplied_key = request.headers.get("X-Monitoring-Key", "")
    if configured_key and not secrets.compare_digest(supplied_key, configured_key):
        return JsonResponse({"error": "forbidden"}, status=403)
    try:
        days = int(request.GET.get("days", 7))
    except ValueError:
        return JsonResponse({"error": "days must be an integer"}, status=400)
    return JsonResponse(monitoring_summary(days=days))
