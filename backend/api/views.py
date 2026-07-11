import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .services import (
    PredictionInputError, estimate_details, harris_estimate, harris_metadata,
    market_context, model_metadata,
)


logger = logging.getLogger(__name__)


def model_info(request):
    """Return training metrics and global feature importance for the UI."""
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)
    metadata = model_metadata()
    return JsonResponse({
        "deployed_model": metadata["deployed_model"],
        "dataset_rows": metadata["dataset_rows"],
        "training_rows": metadata["training_rows"],
        "test_rows": metadata["test_rows"],
        "metrics": metadata["gradient_boost"],
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
    try:
        payload = json.loads(request.body)
        estimate = harris_estimate(payload)
    except (TypeError, json.JSONDecodeError):
        return JsonResponse({"error": "Request body must contain valid JSON."}, status=400)
    except PredictionInputError as error:
        return JsonResponse({"error": str(error)}, status=400)
    except Exception:
        logger.exception("Harris County prediction failed")
        return JsonResponse({"error": "Harris County prediction service is unavailable."}, status=503)
    return JsonResponse({
        "prediction": round(estimate["prediction"], 2),
        "range": {key: round(value, 2) for key, value in estimate["range"].items()},
        "confidence": estimate["confidence"], "disclaimer": estimate["disclaimer"],
    })


@require_POST
@csrf_exempt
def predict(request):
    """Return a house-price estimate for a validated JSON request."""
    try:
        payload = json.loads(request.body)
    except (TypeError, json.JSONDecodeError):
        return JsonResponse({"error": "Request body must contain valid JSON."}, status=400)

    try:
        estimate = estimate_details(payload)
    except PredictionInputError as error:
        return JsonResponse({"error": str(error)}, status=400)
    except Exception:
        logger.exception("House-price prediction failed")
        return JsonResponse({"error": "Prediction service is unavailable."}, status=503)

    result = {
        "prediction": round(estimate["prediction"], 2),
        "range": {key: round(value, 2) for key, value in estimate["range"].items()},
        "confidence": estimate["confidence"],
        "disclaimer": "Experimental estimate based on historical Ames, Iowa-style housing data; not a formal appraisal.",
    }
    if payload.get("address"):
        result["market"] = market_context(payload["address"])
    return JsonResponse(result)
