"""Privacy-safe preparation of King County residential sale records.

Only model-relevant columns are read from the source archives. Owner names,
street addresses, recording numbers, and other direct identifiers never enter
the prepared dataframe.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "king-county"
OUTPUT_PATH = ROOT / "data" / "processed" / "king-county" / "sales.csv"
AUDIT_PATH = ROOT / "reports" / "king_county_data_audit.json"

ARCHIVES = {
    "sales": ("Real Property Sales.zip", "EXTR_RPSale.csv"),
    "buildings": ("Residential Building.zip", "EXTR_ResBldg.csv"),
    "parcels": ("Parcel.zip", "EXTR_Parcel.csv"),
}

SALES_COLUMNS = [
    "Major", "Minor", "DocumentDate", "SalePrice", "PropertyType",
    "PrincipalUse", "SaleInstrument", "SaleReason", "PropertyClass", "SaleWarning",
]
BUILDING_COLUMNS = [
    "Major", "Minor", "ZipCode", "Stories", "BldgGrade", "SqFtTotLiving",
    "SqFtFinBasement", "SqFtGarageBasement", "SqFtGarageAttached", "Bedrooms",
    "BathHalfCount", "Bath3qtrCount", "BathFullCount", "FpSingleStory",
    "FpMultiStory", "FpFreestanding", "FpAdditional", "YrBuilt", "YrRenovated",
    "Condition", "HeatSystem", "ViewUtilization",
]
PARCEL_COLUMNS = ["Major", "Minor", "PresentUse", "SqFtLot"]

MODEL_FEATURES = [
    "beds", "baths", "size", "lot_size", "zip_code", "stories", "grade",
    "condition", "year_built", "year_renovated", "finished_basement_sqft",
    "garage_sqft", "fireplaces", "heat_system", "has_view", "property_type",
    "sale_year", "sale_month",
]


def _read_archive(path: Path, member: str, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing King County archive: {path}")
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [member]:
            raise ValueError(f"Unexpected contents in {path.name}: {archive.namelist()}")
    return pd.read_csv(
        path,
        compression="zip",
        encoding="cp1252",
        usecols=columns,
        dtype={"Major": "string", "Minor": "string"},
        low_memory=False,
    )


def _parcel_key(frame: pd.DataFrame) -> pd.Series:
    major = frame["Major"].astype("string").str.strip().str.zfill(6)
    minor = frame["Minor"].astype("string").str.strip().str.zfill(4)
    return major + minor


def prepare_frames(
    sales: pd.DataFrame,
    buildings: pd.DataFrame,
    parcels: pd.DataFrame,
    *,
    start_year: int = 2015,
    end_year: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Create an arm's-length Seattle-ZIP cohort and an auditable filter summary."""
    sales = sales.copy()
    buildings = buildings.copy()
    parcels = parcels.copy()
    counts: dict[str, int] = {"source_sales": len(sales)}

    sales["sale_date"] = pd.to_datetime(sales["DocumentDate"], errors="coerce")
    sale_price = pd.to_numeric(sales["SalePrice"], errors="coerce")
    sale_year = sales["sale_date"].dt.year
    valid = sale_year.ge(start_year) & sale_price.gt(0)
    if end_year is not None:
        valid &= sale_year.le(end_year)
    filters: Mapping[str, pd.Series] = {
        "residential_principal_use": pd.to_numeric(sales["PrincipalUse"], errors="coerce").eq(6),
        "improved_residential_class": pd.to_numeric(sales["PropertyClass"], errors="coerce").eq(8),
        "normal_sale_reason": pd.to_numeric(sales["SaleReason"], errors="coerce").eq(1),
        "warranty_deed": pd.to_numeric(sales["SaleInstrument"], errors="coerce").isin([2, 3]),
        "no_sale_warning": sales["SaleWarning"].fillna("").astype(str).str.strip().eq(""),
    }
    counts["valid_date_and_positive_price"] = int(valid.sum())
    for label, condition in filters.items():
        valid &= condition
        counts[f"after_{label}"] = int(valid.sum())
    sales = sales.loc[valid].copy()
    sales["parcel_id"] = _parcel_key(sales)
    sales["price"] = sale_price.loc[valid].astype(float)
    sales["sale_year"] = sales["sale_date"].dt.year.astype(int)
    sales["sale_month"] = sales["sale_date"].dt.month.astype(int)

    buildings["parcel_id"] = _parcel_key(buildings)
    buildings["SqFtTotLiving"] = pd.to_numeric(buildings["SqFtTotLiving"], errors="coerce")
    buildings = (
        buildings.sort_values(["parcel_id", "SqFtTotLiving"], ascending=[True, False])
        .drop_duplicates("parcel_id", keep="first")
    )
    parcels["parcel_id"] = _parcel_key(parcels)
    parcels = parcels.drop_duplicates("parcel_id", keep="first")

    joined = sales.merge(buildings.drop(columns=["Major", "Minor"]), on="parcel_id", how="inner")
    counts["after_building_join"] = len(joined)
    joined = joined.merge(parcels.drop(columns=["Major", "Minor"]), on="parcel_id", how="inner")
    counts["after_parcel_join"] = len(joined)

    numeric_sources = set(BUILDING_COLUMNS + PARCEL_COLUMNS) - {"Major", "Minor", "ViewUtilization"}
    for column in numeric_sources:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    joined["ZipCode"] = pd.to_numeric(joined["ZipCode"], errors="coerce")
    valid_property = (
        joined["ZipCode"].between(98101, 98199)
        & joined["SqFtTotLiving"].gt(0)
        & joined["SqFtLot"].gt(0)
        & joined["YrBuilt"].between(1800, joined["sale_year"])
        & joined["Bedrooms"].ge(0)
    )
    joined = joined.loc[valid_property].copy()
    counts["after_property_validation_and_seattle_zip"] = len(joined)

    baths = joined["BathFullCount"] + 0.75 * joined["Bath3qtrCount"] + 0.5 * joined["BathHalfCount"]
    fireplaces = joined[["FpSingleStory", "FpMultiStory", "FpFreestanding", "FpAdditional"]].sum(axis=1)
    output = pd.DataFrame({
        "parcel_id": joined["parcel_id"],
        "sale_date": joined["sale_date"].dt.strftime("%Y-%m-%d"),
        "price": joined["price"],
        "beds": joined["Bedrooms"],
        "baths": baths,
        "size": joined["SqFtTotLiving"],
        "lot_size": joined["SqFtLot"],
        "zip_code": joined["ZipCode"].astype(int).astype(str),
        "stories": joined["Stories"],
        "grade": joined["BldgGrade"],
        "condition": joined["Condition"],
        "year_built": joined["YrBuilt"],
        "year_renovated": joined["YrRenovated"].replace(0, np.nan),
        "finished_basement_sqft": joined["SqFtFinBasement"],
        "garage_sqft": joined["SqFtGarageBasement"] + joined["SqFtGarageAttached"],
        "fireplaces": fireplaces,
        "heat_system": joined["HeatSystem"].astype("Int64").astype("string"),
        "has_view": joined["ViewUtilization"].fillna("N").astype(str).str.upper().eq("Y").astype(int),
        "property_type": joined["PresentUse"].astype("Int64").astype("string"),
        "sale_year": joined["sale_year"],
        "sale_month": joined["sale_month"],
    })
    duplicate_rows_removed = int(output.duplicated().sum())
    counts["before_exact_duplicate_removal"] = len(output)
    output = output.drop_duplicates().copy()
    output = output.sort_values(["sale_date", "parcel_id"]).reset_index(drop=True)
    counts["output_rows"] = len(output)
    audit = {
        "source": "King County Assessor public extracts",
        "privacy": "Owner names, street addresses, and recording identifiers were not read.",
        "filter_counts": counts,
        "rows": len(output),
        "unique_parcels": int(output["parcel_id"].nunique()),
        "exact_duplicate_rows_removed": duplicate_rows_removed,
        "duplicate_rows_remaining": int(output.duplicated().sum()),
        "duplicate_sale_keys": int(output.duplicated(["parcel_id", "sale_date", "price"]).sum()),
        "date_min": output["sale_date"].min(),
        "date_max": output["sale_date"].max(),
        "columns": output.columns.tolist(),
        "price_quantiles": {str(k): float(v) for k, v in output["price"].quantile([0, .01, .5, .99, 1]).items()},
        "missing_values": {name: int(value) for name, value in output.isna().sum().items()},
        "suspicious_values_retained_for_audit": {
            "price_below_100k": int(output["price"].lt(100_000).sum()),
            "living_area_above_10000": int(output["size"].gt(10_000).sum()),
            "lot_area_above_1m": int(output["lot_size"].gt(1_000_000).sum()),
            "bedrooms_above_10": int(output["beds"].gt(10).sum()),
        },
        "cleaning_rules": [
            f"Document year is at least {start_year}" + (f" and at most {end_year}" if end_year else ""),
            "Sale price is positive",
            "PrincipalUse=6 and PropertyClass=8",
            "SaleReason=1, SaleInstrument in {2,3}, and SaleWarning is blank",
            "Largest residential building record per parcel is selected",
            "ZIP is 98101-98199; living area and lot area are positive",
            "Year built is between 1800 and sale year; bedrooms are non-negative",
            "Exact duplicate prepared rows are removed; no fuzzy deduplication is applied",
            "No price-quantile outlier trimming is applied",
        ],
    }
    return output, audit


def prepare_dataset(raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH) -> tuple[pd.DataFrame, dict]:
    frames = {
        name: _read_archive(raw_dir / archive, member, columns)
        for name, (archive, member), columns in (
            ("sales", ARCHIVES["sales"], SALES_COLUMNS),
            ("buildings", ARCHIVES["buildings"], BUILDING_COLUMNS),
            ("parcels", ARCHIVES["parcels"], PARCEL_COLUMNS),
        )
    }
    output, audit = prepare_frames(frames["sales"], frames["buildings"], frames["parcels"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return output, audit


if __name__ == "__main__":
    prepared, summary = prepare_dataset()
    print(f"Prepared {len(prepared):,} rows at {OUTPUT_PATH.relative_to(ROOT)}")
    print(json.dumps(summary["filter_counts"], indent=2))
