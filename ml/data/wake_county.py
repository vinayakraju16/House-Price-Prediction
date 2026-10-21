"""Privacy-safe preparation of Wake County, NC residential sale records.

Two public files from Wake County Tax Administration are combined:

* ``Qualified_Sales_Past_24Months.xlsx`` - sales an appraiser reviewed as likely
  arm's-length (foreclosures, related-party and partial-interest sales are already
  excluded upstream). One row per parcel per deed; ``*_ATSALE`` columns describe the
  parcel when it sold, the rest describe it today.
* ``RealEstDataMMDDYYYY.zip`` / ``.txt`` - the full fixed-width real estate file
  (711 characters per line, one line per building "card"). It adds construction
  details the sales file lacks: baths, grade, heating, A/C, basement, remodel year.

Only model-relevant columns are read. Owner names, mailing addresses, and street
addresses (present in both files) never enter the prepared dataframe: the sales
workbook is read with ``usecols`` and the fixed-width file with explicit column specs
that skip positions 1-175 (owners/mailing) and 189-234 (site address).

Wake County publishes no bedroom count, so this market has no ``beds`` feature.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "wake-county"
OUTPUT_PATH = ROOT / "data" / "processed" / "wake-county" / "sales.csv"
AUDIT_PATH = ROOT / "reports" / "wake_county_data_audit.json"

SALES_FILE = "Qualified_Sales_Past_24Months.xlsx"
FULL_FILE_PATTERN = re.compile(r"^RealEstData(\d{2})(\d{2})(\d{4})\.(zip|txt)$", re.IGNORECASE)
FULL_FILE_LINE_LENGTH = 711

SALES_COLUMNS = [
    "REID", "SALE_DEED_BOOK", "SALE_DEED_PAGE", "SALE_PRICE", "SALE_DATE", "SALE_TYPE",
    "PARCEL_DEEDED_ACREAGE_ATSALE", "TOTAL_HEATED_AREA_ATSALE",
    "PARCEL_DEEDED_ACREAGE_CURRENT", "TOTAL_HEATED_AREA_CURRENT",
    "BLDG_USE_DESC", "YEAR_BUILT", "DESIGN_STYLE_DESC", "ZIP",
]

# (name, 1-based start, width) from Wake County's "Record Layout" document.
# Owner, mailing-address, and street-address positions are deliberately absent.
FULL_FILE_FIELDS = [
    ("REID", 176, 7),
    ("CARD", 183, 3),
    ("NUM_CARDS", 186, 3),
    ("YEAR_BUILT", 456, 4),
    ("UNITS", 466, 6),
    ("HEATED_AREA", 472, 11),
    ("YEAR_OF_ADDITION", 488, 4),
    ("EFFECTIVE_YEAR", 492, 4),
    ("REMODELED_YEAR", 496, 4),
    ("STORY_HEIGHT", 510, 1),
    ("DESIGN_STYLE", 511, 1),
    ("FOUNDATION_BASEMENT", 512, 1),
    ("EXTERIOR_WALL", 515, 1),
    ("HEAT", 527, 1),
    ("AIR", 530, 1),
    ("BATH", 533, 1),
    ("BATH_FIXTURES", 534, 3),
    ("GRADE", 615, 5),
    ("CONDITION_PCT", 623, 3),
    ("TYPE_AND_USE", 654, 3),
    ("PHYSICAL_ZIP", 707, 5),
]

# Code tables from Wake County's "Code Descriptions" document.
BATH_CODES = {"A": 1.0, "B": 1.5, "C": 2.0, "D": 2.5, "E": 3.0, "F": 3.5}
STORY_CODES = {
    "A": 1.0, "B": 1.5, "C": 2.0, "D": 2.5, "E": 3.0, "F": 3.5, "G": 4.0,
    "I": 1.75, "J": 1.4, "K": 1.63, "L": 1.88, "M": 2.4, "N": 2.63, "R": 2.75,
}
GRADE_BASE = {"AA": 6.0, "A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}
GRADE_PATTERN = re.compile(r"^(AA|A|B|C|D|E)(?:([+-])(\d{1,2}))?$")
RESIDENTIAL_TYPE_AND_USE = {1, 8}  # one family, townhouse
SQFT_PER_ACRE = 43_560

MODEL_FEATURES = [
    "baths", "size", "lot_size", "zip_code", "stories", "grade", "condition",
    "year_built", "year_renovated", "has_basement", "has_central_air", "heat_system",
    "exterior_wall", "property_type", "sale_year", "sale_month",
]


def _read_sales(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Wake County qualified sales workbook: {path}")
    return pd.read_excel(
        path,
        engine="openpyxl",
        usecols=SALES_COLUMNS,
        dtype={"REID": "string", "SALE_DEED_BOOK": "string", "SALE_DEED_PAGE": "string",
               "ZIP": "string"},
    )


def find_full_file(raw_dir: Path = RAW_DIR) -> Path:
    """Return the newest ``RealEstDataMMDDYYYY`` .zip/.txt (date taken from the name)."""
    candidates = []
    for path in raw_dir.glob("RealEstData*"):
        match = FULL_FILE_PATTERN.match(path.name)
        if match:
            month, day, year, suffix = match.groups()
            candidates.append(((year, month, day, suffix.lower() == "txt"), path))
    if not candidates:
        raise FileNotFoundError(f"No RealEstDataMMDDYYYY.zip or .txt found in {raw_dir}")
    return max(candidates)[1]


def _read_full_file(path: Path) -> pd.DataFrame:
    colspecs = [(start - 1, start - 1 + width) for _, start, width in FULL_FILE_FIELDS]
    names = [name for name, _, _ in FULL_FILE_FIELDS]
    options = dict(colspecs=colspecs, names=names, header=None, dtype="string",
                   encoding="cp1252")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith(".txt")]
            if len(members) != 1:
                raise ValueError(f"Expected one .txt in {path.name}, found {archive.namelist()}")
            with archive.open(members[0]) as handle:
                _check_line_length(handle.readline(), path)
            with archive.open(members[0]) as handle:
                return pd.read_fwf(io.TextIOWrapper(handle, encoding="cp1252"),
                                   **{k: v for k, v in options.items() if k != "encoding"})
    with path.open("rb") as handle:
        _check_line_length(handle.readline(), path)
    return pd.read_fwf(path, **options)


def _check_line_length(first_line: bytes, path: Path) -> None:
    length = len(first_line.rstrip(b"\r\n"))
    if length != FULL_FILE_LINE_LENGTH:
        raise ValueError(
            f"{path.name}: first record is {length} characters, expected "
            f"{FULL_FILE_LINE_LENGTH}. The Wake County record layout may have changed."
        )


def _reid(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(7)


def _sale_dates(series: pd.Series) -> pd.Series:
    """Excel serials (raw workbook values) or already-parsed datetimes."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() and numeric.notna().sum() >= series.notna().sum():
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def grade_score(code: object) -> float:
    """Map a residential grade such as ``C+10`` or ``B-05`` to a number.

    AA=6 ... E=1, and the +/- percentage modifier shifts it by hundredths, so
    ``C+10`` -> 3.10 and ``B-05`` -> 3.95. Commercial numeric grades return NaN.
    """
    if code is None or pd.isna(code):
        return np.nan
    match = GRADE_PATTERN.match(str(code).strip().upper())
    if not match:
        return np.nan
    base, sign, amount = match.groups()
    shift = (int(amount) / 100) * (1 if sign == "+" else -1) if sign else 0.0
    return GRADE_BASE[base] + shift


def prepare_frames(
    sales: pd.DataFrame,
    buildings: pd.DataFrame,
    *,
    start_year: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Create an arm's-length Wake County single-family cohort and an auditable summary."""
    sales = sales.copy()
    buildings = buildings.copy()
    counts: dict[str, int] = {"source_sales": len(sales), "source_building_cards": len(buildings)}

    sales["sale_date"] = _sale_dates(sales["SALE_DATE"])
    sales["price"] = pd.to_numeric(sales["SALE_PRICE"], errors="coerce")
    valid = sales["sale_date"].notna() & sales["price"].gt(0)
    if start_year is not None:
        valid &= sales["sale_date"].dt.year.ge(start_year)
    counts["valid_date_and_positive_price"] = int(valid.sum())

    deed_key = (
        sales["SALE_DEED_BOOK"].astype("string").str.strip() + "/"
        + sales["SALE_DEED_PAGE"].astype("string").str.strip() + "/"
        + sales["sale_date"].astype("string")
    )
    parcels_per_deed = deed_key.map(deed_key.value_counts())
    heated_at_sale = pd.to_numeric(sales["TOTAL_HEATED_AREA_ATSALE"], errors="coerce")
    heated_now = pd.to_numeric(sales["TOTAL_HEATED_AREA_CURRENT"], errors="coerce")
    acres_at_sale = pd.to_numeric(sales["PARCEL_DEEDED_ACREAGE_ATSALE"], errors="coerce")
    acres_now = pd.to_numeric(sales["PARCEL_DEEDED_ACREAGE_CURRENT"], errors="coerce")
    filters = {
        "improved_sale": sales["SALE_TYPE"].astype("string").str.strip().str.lower().eq("package"),
        "single_family_use": sales["BLDG_USE_DESC"].astype("string").str.strip().eq("SINGLFAM"),
        "single_parcel_deed": parcels_per_deed.eq(1),
        "positive_heated_area_and_acreage": heated_at_sale.gt(0) & acres_at_sale.gt(0),
        # Current characteristics are only valid for the sale if nothing changed since.
        "characteristics_unchanged_since_sale": heated_at_sale.eq(heated_now)
        & acres_at_sale.eq(acres_now),
    }
    for label, condition in filters.items():
        valid &= condition.fillna(False)
        counts[f"after_{label}"] = int(valid.sum())
    sales = sales.loc[valid].copy()
    sales["parcel_id"] = _reid(sales["REID"])
    sales["size"] = heated_at_sale.loc[valid]
    sales["lot_size"] = (acres_at_sale.loc[valid] * SQFT_PER_ACRE).round()

    buildings["parcel_id"] = _reid(buildings["REID"])
    for column in ("CARD", "NUM_CARDS", "HEATED_AREA", "TYPE_AND_USE", "YEAR_BUILT",
                   "REMODELED_YEAR", "CONDITION_PCT"):
        buildings[column] = pd.to_numeric(buildings[column], errors="coerce")
    single_card = buildings["NUM_CARDS"].eq(1) & buildings["CARD"].eq(1)
    counts["single_card_parcels"] = int(single_card.sum())
    buildings = buildings.loc[single_card].drop_duplicates("parcel_id", keep="first")

    joined = sales.merge(buildings, on="parcel_id", how="inner", suffixes=("", "_full"))
    counts["after_single_card_full_file_join"] = len(joined)

    joined["sale_year"] = joined["sale_date"].dt.year.astype(int)
    joined["sale_month"] = joined["sale_date"].dt.month.astype(int)
    zip_code = joined["ZIP"].astype("string").str.strip().str[:5]
    zip_code = zip_code.where(zip_code.str.fullmatch(r"27\d{3}").fillna(False),
                              joined["PHYSICAL_ZIP"].astype("string").str.strip())
    joined["zip_code"] = zip_code
    year_built = pd.to_numeric(joined["YEAR_BUILT"], errors="coerce")
    valid_property = (
        joined["zip_code"].str.fullmatch(r"27\d{3}").fillna(False)
        & joined["TYPE_AND_USE"].isin(RESIDENTIAL_TYPE_AND_USE)
        & year_built.between(1800, joined["sale_year"])
    )
    joined = joined.loc[valid_property].copy()
    counts["after_property_validation_and_wake_zip"] = len(joined)

    bath_code = joined["BATH"].astype("string").str.strip().str.upper()
    remodeled = pd.to_numeric(joined["REMODELED_YEAR"], errors="coerce")
    output = pd.DataFrame({
        "parcel_id": joined["parcel_id"],
        "sale_date": joined["sale_date"].dt.strftime("%Y-%m-%d"),
        "price": joined["price"].astype(float),
        "baths": bath_code.map(BATH_CODES).astype(float),
        "size": joined["size"].astype(float),
        "lot_size": joined["lot_size"].astype(float),
        "zip_code": joined["zip_code"].astype(str),
        "stories": joined["STORY_HEIGHT"].astype("string").str.strip().str.upper()
        .map(STORY_CODES).astype(float),
        "grade": joined["GRADE"].map(grade_score).astype(float),
        "condition": joined["CONDITION_PCT"].astype(float),
        "year_built": year_built.loc[joined.index].astype(float),
        "year_renovated": remodeled.where(remodeled.gt(0)).astype(float),
        "has_basement": joined["FOUNDATION_BASEMENT"].astype("string").str.strip().str.upper()
        .isin(["A", "B"]).astype(int),
        "has_central_air": joined["AIR"].astype("string").str.strip().str.upper()
        .isin(["A", "B", "D", "E"]).astype(int),
        "heat_system": joined["HEAT"].astype("string").str.strip().str.upper(),
        "exterior_wall": joined["EXTERIOR_WALL"].astype("string").str.strip().str.upper(),
        "property_type": joined["DESIGN_STYLE_DESC"].astype("string").str.strip(),
        "sale_year": joined["sale_year"],
        "sale_month": joined["sale_month"],
    })
    duplicate_rows_removed = int(output.duplicated().sum())
    counts["before_exact_duplicate_removal"] = len(output)
    output = output.drop_duplicates().sort_values(["sale_date", "parcel_id"]).reset_index(drop=True)
    counts["output_rows"] = len(output)

    audit = {
        "source": "Wake County Tax Administration qualified sales and real estate data file",
        "privacy": "Owner names, mailing addresses, and street addresses were not read.",
        "filter_counts": counts,
        "rows": len(output),
        "unique_parcels": int(output["parcel_id"].nunique()),
        "exact_duplicate_rows_removed": duplicate_rows_removed,
        "duplicate_sale_keys": int(output.duplicated(["parcel_id", "sale_date", "price"]).sum()),
        "date_min": output["sale_date"].min() if len(output) else None,
        "date_max": output["sale_date"].max() if len(output) else None,
        "columns": output.columns.tolist(),
        "price_quantiles": {
            str(k): float(v) for k, v in output["price"].quantile([0, .01, .5, .99, 1]).items()
        } if len(output) else {},
        "missing_values": {name: int(value) for name, value in output.isna().sum().items()},
        "suspicious_values_retained_for_audit": {
            "price_below_100k": int(output["price"].lt(100_000).sum()),
            "living_area_above_10000": int(output["size"].gt(10_000).sum()),
            "lot_area_above_1m": int(output["lot_size"].gt(1_000_000).sum()),
        },
        "known_gaps": [
            "No bedroom count is published by Wake County.",
            "Bath code F means 3.5 or more baths; codes G-J (limited/adequate/fixtures) are NaN.",
            "No garage, fireplace, or view fields are published.",
        ],
        "cleaning_rules": [
            "Rows come from the county's qualified (arm's-length) sales list",
            "SALE_TYPE is Package (land and building), not Land-only",
            "BLDG_USE_DESC is SINGLFAM; full-file Type & Use is 1 (one family) or 8 (townhouse)",
            "Deeds covering more than one parcel are removed (price is a package total)",
            "Heated area and acreage at sale equal current values (characteristics still valid)",
            "Only parcels with exactly one building card are kept",
            "ZIP is a 27xxx Wake County ZIP; year built is between 1800 and sale year",
            "Exact duplicate prepared rows are removed; no price-quantile trimming is applied",
        ],
    }
    return output, audit


def prepare_dataset(
    raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH
) -> tuple[pd.DataFrame, dict]:
    sales_path = raw_dir / SALES_FILE
    full_path = find_full_file(raw_dir)
    output, audit = prepare_frames(_read_sales(sales_path), _read_full_file(full_path))
    audit["source_files"] = [sales_path.name, full_path.name]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return output, audit


if __name__ == "__main__":
    prepared, summary = prepare_dataset()
    print(f"Prepared {len(prepared):,} rows at {OUTPUT_PATH.relative_to(ROOT)}")
    print(json.dumps(summary["filter_counts"], indent=2))
