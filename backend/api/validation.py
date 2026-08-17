"""Request validation for property prediction endpoints."""

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

FEATURE_NAMES = ("beds", "baths", "size", "lot_size", "zip_code")
OPTIONAL_FEATURE_NAMES = (
    "stories", "grade", "condition", "year_built", "year_renovated",
    "finished_basement_sqft", "garage_sqft", "fireplaces", "heat_system",
    "has_view", "property_type",
)
INTEGER_FEATURES = {
    "beds", "zip_code", "grade", "condition", "year_built", "year_renovated",
    "fireplaces", "heat_system", "has_view", "property_type",
}


class PredictionInputError(ValueError):
    """A request validation failure with machine-readable field details."""

    def __init__(self, message: str, *, details: dict[str, str] | None = None):
        super().__init__(message)
        self.details = details or {"request": message}


def validate_prediction_payload(payload: Any) -> dict[str, int | float]:
    """Validate required and optional model fields without mutating input."""
    if not isinstance(payload, Mapping):
        message = "The request body must be a JSON object."
        raise PredictionInputError(message, details={"body": message})

    details: dict[str, str] = {}
    values: dict[str, int | float] = {}
    for name in FEATURE_NAMES:
        raw_value = payload.get(name)
        if raw_value in (None, ""):
            details[name] = "This field is required."
            continue

        if name == "zip_code":
            normalized = _validate_zip_code(raw_value, details)
        else:
            normalized = _validate_numeric_feature(name, raw_value, details)
        if normalized is not None:
            values[name] = normalized

    for name in OPTIONAL_FEATURE_NAMES:
        raw_value = payload.get(name)
        if raw_value in (None, ""):
            continue
        normalized = _validate_numeric_feature(name, raw_value, details)
        if normalized is not None:
            values[name] = normalized

    _validate_feature_ranges(values, details)

    if details:
        message = "; ".join(f"{field}: {problem}" for field, problem in details.items())
        raise PredictionInputError(message, details=details)
    return values


def _validate_feature_ranges(values: dict, details: dict[str, str]) -> None:
    current_year = datetime.now(timezone.utc).year
    ranges = {
        "beds": (0, 20), "baths": (0, 20), "size": (1, 30_000),
        "lot_size": (0, 5_000_000), "stories": (0.5, 100), "grade": (1, 13),
        "condition": (1, 5), "year_built": (1800, current_year),
        "year_renovated": (0, current_year), "finished_basement_sqft": (0, 30_000),
        "garage_sqft": (0, 30_000), "fireplaces": (0, 30), "heat_system": (0, 99),
        "has_view": (0, 1), "property_type": (0, 999),
    }
    for name, (minimum, maximum) in ranges.items():
        if name in values and not minimum <= values[name] <= maximum:
            details[name] = f"Must be between {minimum:g} and {maximum:g}."
    if "zip_code" in values and not 98101 <= values["zip_code"] <= 98199:
        details["zip_code"] = "Must be a Seattle-area ZIP code from 98101 to 98199."
    if values.get("year_renovated") == 0:
        values.pop("year_renovated")
    if (
        "year_renovated" in values
        and "year_built" in values
        and values["year_renovated"] < values["year_built"]
    ):
        details["year_renovated"] = "Cannot be earlier than year built."


def _validate_zip_code(value: Any, details: dict[str, str]) -> int | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        details["zip_code"] = "Must be a valid 5-digit ZIP code."
        return None
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        details["zip_code"] = "Must be a whole 5-digit ZIP code."
        return None
    normalized = int(numeric_value)
    if normalized < 10000 or normalized > 99999:
        details["zip_code"] = "Must be a valid 5-digit ZIP code."
        return None
    return normalized


def _validate_numeric_feature(
    name: str, value: Any, details: dict[str, str]
) -> int | float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        details[name] = "Must be a number."
        return None
    if not math.isfinite(normalized):
        details[name] = "Must be a finite number."
    elif name == "size" and normalized <= 0:
        details[name] = "Must be greater than zero."
    elif normalized < 0:
        details[name] = "Cannot be negative."
    elif name in INTEGER_FEATURES and not normalized.is_integer():
        details[name] = "Must be a whole number."
    else:
        return int(normalized) if name in INTEGER_FEATURES else normalized
    return None
