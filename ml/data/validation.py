"""Schema and domain validation for the Seattle listing datasets.

Cleaning policy:
- exact duplicate rows are removed;
- acre-valued size fields are converted to square feet using 43,560;
- a missing lot size is represented as zero because the source omits lot data for
  some unit types; the missingness should remain available to models as a flag;
- price and core property fields must not be silently imputed or discarded;
- unusual but valid luxury properties are reported, not automatically removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

FEATURE_COLUMNS = ("beds", "baths", "size", "lot_size", "zip_code")
TARGET_COLUMN = "price"
RAW_COLUMNS = (*FEATURE_COLUMNS[:3], "size_units", "lot_size", "lot_size_units", "zip_code", TARGET_COLUMN)
CLEAN_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
ALLOWED_UNITS = {"sqft", "acre"}


@dataclass
class DataValidationError(ValueError):
    """Raised when a dataset violates the required schema or domain rules."""

    errors: list[str]

    def __str__(self) -> str:
        return "; ".join(self.errors)


def _native(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def audit_seattle_dataframe(frame: pd.DataFrame, *, stage: str = "raw") -> dict[str, Any]:
    """Return a JSON-serializable validation report without mutating the data."""
    if stage not in {"raw", "cleaned"}:
        raise ValueError("stage must be 'raw' or 'cleaned'")

    expected = list(RAW_COLUMNS if stage == "raw" else CLEAN_COLUMNS)
    missing_columns = [column for column in expected if column not in frame.columns]
    extra_columns = [column for column in frame.columns if column not in expected]
    errors: list[str] = []
    warnings: list[str] = []

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    if extra_columns:
        warnings.append(f"Unexpected columns: {', '.join(extra_columns)}")

    report: dict[str, Any] = {
        "stage": stage,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_by_column": {name: int(count) for name, count in frame.isna().sum().items()},
        "errors": errors,
        "warnings": warnings,
    }
    if report["duplicate_rows"]:
        message = f"Found {report['duplicate_rows']} exact duplicate rows"
        (errors if stage == "cleaned" else warnings).append(message)

    for column in (*FEATURE_COLUMNS, TARGET_COLUMN):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = int((values.isna() & frame[column].notna()).sum())
        non_finite = int(values.map(lambda value: pd.notna(value) and not math.isfinite(float(value))).sum())
        if invalid:
            errors.append(f"{column} contains {invalid} non-numeric values")
        if non_finite:
            errors.append(f"{column} contains {non_finite} non-finite values")

        finite = values[values.map(lambda value: pd.notna(value) and math.isfinite(float(value)))]
        if finite.empty:
            continue
        report.setdefault("ranges", {})[column] = {
            "minimum": float(finite.min()),
            "median": float(finite.median()),
            "maximum": float(finite.max()),
        }
        if column in {"size", TARGET_COLUMN} and bool((finite <= 0).any()):
            errors.append(f"{column} must be greater than zero")
        if column in {"beds", "baths", "lot_size"} and bool((finite < 0).any()):
            errors.append(f"{column} cannot be negative")
        if column in {"beds", "zip_code"} and bool((finite % 1 != 0).any()):
            errors.append(f"{column} must contain whole numbers")
        if column == "zip_code" and bool(((finite < 10000) | (finite > 99999)).any()):
            errors.append("zip_code must contain 5-digit ZIP codes")

    if stage == "raw":
        for column in ("size_units", "lot_size_units"):
            if column not in frame.columns:
                continue
            normalized = set(frame[column].dropna().astype(str).str.strip().str.lower().unique())
            unsupported = sorted(normalized - ALLOWED_UNITS)
            if unsupported:
                errors.append(f"{column} contains unsupported units: {', '.join(unsupported)}")
        if "lot_size" in frame and "lot_size_units" in frame:
            mismatched = frame["lot_size"].isna() != frame["lot_size_units"].isna()
            if bool(mismatched.any()):
                errors.append("lot_size and lot_size_units must be missing together")
    elif any(int(count) for count in report["missing_by_column"].values()):
        errors.append("Cleaned data cannot contain missing values")

    if TARGET_COLUMN in frame.columns:
        target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").dropna()
        if not target.empty:
            report["target"] = {
                "skew": float(target.skew()),
                "p01": float(target.quantile(0.01)),
                "p99": float(target.quantile(0.99)),
                "iqr_outliers_3x": int(_iqr_outlier_mask(target, multiplier=3.0).sum()),
            }

    report["valid"] = not errors
    return {key: _native(value) for key, value in report.items()}


def validate_seattle_dataframe(frame: pd.DataFrame, *, stage: str = "raw") -> dict[str, Any]:
    """Audit a Seattle frame and raise when critical validation rules fail."""
    report = audit_seattle_dataframe(frame, stage=stage)
    if report["errors"]:
        raise DataValidationError(report["errors"])
    return report


def _iqr_outlier_mask(values: pd.Series, *, multiplier: float) -> pd.Series:
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (values < q1 - multiplier * iqr) | (values > q3 + multiplier * iqr)
