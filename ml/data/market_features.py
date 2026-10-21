"""ZIP-month market features shared by every market model.

Sources (all free, national, ZIP level, refreshed monthly or yearly), stored in
``data/raw/market/``:

* ``Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv`` - Zillow Home Value Index,
  typical (mid-tier) home value per ZIP, smoothed and seasonally adjusted.
* ``RDC_Inventory_Core_Metrics_Zip_History.csv`` - Realtor.com listing inventory per ZIP:
  days on market, price-cut share, pending ratio, active listings year over year.
* ``PMMS_history.csv`` - Freddie Mac weekly 30-year fixed mortgage rate.
* ``acs5_2024_zcta.csv`` - Census ACS 2020-2024 5-year estimates per ZIP code tabulation
  area (median household income, owner-occupied share), built from the table-based
  summary files.

Leakage rule: a sale in month *m* only sees market data for month *m-1* or earlier.
Monthly figures are published two to six weeks after the month ends and homes go under
contract a month or more before closing, so month *m-1* is what a buyer could have known.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MARKET_DIR = ROOT / "data" / "raw" / "market"
ZHVI_FILE = "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
INVENTORY_FILE = "RDC_Inventory_Core_Metrics_Zip_History.csv"
MORTGAGE_FILE = "PMMS_history.csv"
ACS_FILE = "acs5_2024_zcta.csv"
LAG_MONTHS = 1
SMOOTHING_MONTHS = 3

ZHVI_FEATURES = ["zhvi", "zhvi_yoy"]
INVENTORY_FEATURES = [
    "days_on_market", "price_reduced_share", "inventory_yoy", "pending_ratio",
]
RATE_FEATURES = ["mortgage_rate_30y", "mortgage_rate_change_1y"]
ACS_FEATURES = ["median_household_income", "owner_occupied_share"]
MARKET_FEATURES = [*ZHVI_FEATURES, *INVENTORY_FEATURES, *RATE_FEATURES, *ACS_FEATURES]


def _zip5(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return text.str.zfill(5)


def _month_index(series: pd.Series) -> pd.Series:
    """Months since year 0, so 'one month earlier' is simple integer arithmetic."""
    dates = pd.to_datetime(series, errors="coerce")
    return (dates.dt.year * 12 + dates.dt.month - 1).astype("Int64")


def load_zhvi(path: Path, zip_codes: Iterable[str] | None = None) -> pd.DataFrame:
    """Long table: zip_code, month_index, zhvi."""
    header = pd.read_csv(path, nrows=0).columns
    month_columns = [column for column in header if column[:4].isdigit()]
    wide = pd.read_csv(path, usecols=["RegionName", *month_columns],
                       dtype={"RegionName": str})
    wide.index = pd.Index(_zip5(wide.pop("RegionName")), name="zip_code")
    if zip_codes is not None:
        wide = wide.loc[wide.index.isin(set(zip_codes))]
    long = wide.copy().reset_index().melt(id_vars="zip_code", var_name="month",
                                          value_name="zhvi")
    long["month_index"] = _month_index(long["month"])
    return long.dropna(subset=["zhvi"])[["zip_code", "month_index", "zhvi"]]


def load_inventory(path: Path, zip_codes: Iterable[str] | None = None) -> pd.DataFrame:
    """Long table of Realtor.com listing metrics, smoothed over three months."""
    columns = {
        "month_date_yyyymm": "month", "postal_code": "zip_code",
        "median_days_on_market": "days_on_market",
        "price_reduced_share": "price_reduced_share",
        "active_listing_count_yy": "inventory_yoy", "pending_ratio": "pending_ratio",
    }
    frame = pd.read_csv(path, usecols=list(columns), dtype={"postal_code": str})
    frame = frame.rename(columns=columns)
    frame["zip_code"] = _zip5(frame["zip_code"])
    if zip_codes is not None:
        frame = frame.loc[frame["zip_code"].isin(set(zip_codes))]
    month = pd.to_numeric(frame["month"], errors="coerce")
    frame["month_index"] = ((month // 100) * 12 + (month % 100) - 1).astype("Int64")
    # Thin ZIPs swing wildly month to month; clip implausible ratios before smoothing.
    frame["inventory_yoy"] = frame["inventory_yoy"].clip(-1, 3)
    frame["pending_ratio"] = frame["pending_ratio"].clip(0, 5)
    frame["price_reduced_share"] = frame["price_reduced_share"].clip(0, 1)
    frame["days_on_market"] = frame["days_on_market"].clip(0, 365)
    frame = frame.sort_values(["zip_code", "month_index"]).drop_duplicates(
        ["zip_code", "month_index"], keep="last"
    )
    frame = _complete_months(frame, INVENTORY_FEATURES)
    smoothed = (
        frame.groupby("zip_code", sort=False)[INVENTORY_FEATURES]
        .rolling(SMOOTHING_MONTHS, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    frame[INVENTORY_FEATURES] = smoothed
    return frame[["zip_code", "month_index", *INVENTORY_FEATURES]]


def _complete_months(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Insert missing ZIP-months so rolling windows span calendar months, not rows."""
    frame = frame.dropna(subset=["month_index"]).astype({"month_index": "int64"})
    if frame.empty:
        return frame[["zip_code", "month_index", *columns]]
    bounds = frame.groupby("zip_code", sort=True)["month_index"].agg(["min", "max"])
    lengths = (bounds["max"] - bounds["min"] + 1).to_numpy()
    starts = np.repeat(bounds["min"].to_numpy(), lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    full = pd.DataFrame({
        "zip_code": np.repeat(bounds.index.to_numpy(), lengths),
        "month_index": starts + offsets,
    })
    result = full.merge(frame[["zip_code", "month_index", *columns]],
                        on=["zip_code", "month_index"], how="left")
    result["month_index"] = result["month_index"].astype("Int64")
    return result


def load_mortgage_rates(path: Path) -> pd.DataFrame:
    """Monthly mean of Freddie Mac's weekly 30-year rate, plus its 12-month change."""
    frame = pd.read_csv(path, usecols=["date", "pmms30"])
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y", errors="coerce")
    frame["pmms30"] = pd.to_numeric(frame["pmms30"], errors="coerce")
    frame["month_index"] = _month_index(frame["date"])
    monthly = (
        frame.dropna(subset=["pmms30", "month_index"])
        .groupby("month_index", as_index=False)["pmms30"].mean()
        .rename(columns={"pmms30": "mortgage_rate_30y"})
        .sort_values("month_index")
    )
    previous = monthly[["month_index", "mortgage_rate_30y"]].copy()
    previous["month_index"] = previous["month_index"] + 12
    monthly = monthly.merge(previous, on="month_index", how="left", suffixes=("", "_prior"))
    monthly["mortgage_rate_change_1y"] = (
        monthly["mortgage_rate_30y"] - monthly["mortgage_rate_30y_prior"]
    )
    return monthly[["month_index", *RATE_FEATURES]]


def load_acs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"zcta": str})
    frame["zip_code"] = _zip5(frame["zcta"])
    for column in ("median_household_income", "occupied_units", "owner_occupied_units"):
        # The summary files use large negative sentinels for suppressed estimates.
        frame[column] = pd.to_numeric(frame[column], errors="coerce").where(lambda s: s >= 0)
    frame["owner_occupied_share"] = np.divide(
        frame["owner_occupied_units"], frame["occupied_units"],
        out=np.full(len(frame), np.nan), where=frame["occupied_units"].gt(0).to_numpy(),
    )
    return frame[["zip_code", *ACS_FEATURES]]


@dataclass
class MarketData:
    zhvi: pd.DataFrame | None = None
    inventory: pd.DataFrame | None = None
    rates: pd.DataFrame | None = None
    acs: pd.DataFrame | None = None

    @classmethod
    def load(cls, market_dir: Path = MARKET_DIR, zip_codes: Iterable[str] | None = None,
             sources: Iterable[str] = ("zhvi", "inventory", "rates", "acs")) -> MarketData:
        zip_codes = None if zip_codes is None else sorted({str(z).zfill(5) for z in zip_codes})
        sources = set(sources)
        return cls(
            zhvi=load_zhvi(market_dir / ZHVI_FILE, zip_codes) if "zhvi" in sources else None,
            inventory=load_inventory(market_dir / INVENTORY_FILE, zip_codes)
            if "inventory" in sources else None,
            rates=load_mortgage_rates(market_dir / MORTGAGE_FILE) if "rates" in sources else None,
            acs=load_acs(market_dir / ACS_FILE) if "acs" in sources else None,
        )


def add_market_features(
    frame: pd.DataFrame, market: MarketData, *, date_column: str = "sale_date",
    zip_column: str = "zip_code", lag_months: int = LAG_MONTHS,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with every available market feature column added.

    Features for a row dated in month *m* come from month *m - lag_months*; ZHVI momentum
    compares that month with the same month a year earlier. Missing sources or months
    leave NaN, which the model pipelines impute.
    """
    result = frame.copy()
    keys = pd.DataFrame({
        "zip_code": _zip5(result[zip_column]),
        "month_index": _month_index(result[date_column]) - lag_months,
    }, index=result.index)
    if market.zhvi is not None:
        current = keys.merge(market.zhvi, on=["zip_code", "month_index"], how="left")["zhvi"]
        year_ago = keys.assign(month_index=keys["month_index"] - 12).merge(
            market.zhvi, on=["zip_code", "month_index"], how="left"
        )["zhvi"]
        result["zhvi"] = current.to_numpy()
        result["zhvi_yoy"] = (current / year_ago - 1).to_numpy()
    if market.inventory is not None:
        joined = keys.merge(market.inventory, on=["zip_code", "month_index"], how="left")
        for column in INVENTORY_FEATURES:
            result[column] = joined[column].to_numpy()
    if market.rates is not None:
        joined = keys[["month_index"]].merge(market.rates, on="month_index", how="left")
        for column in RATE_FEATURES:
            result[column] = joined[column].to_numpy()
    if market.acs is not None:
        joined = keys[["zip_code"]].merge(market.acs, on="zip_code", how="left")
        for column in ACS_FEATURES:
            result[column] = joined[column].to_numpy()
    return result


def latest_snapshot(market: MarketData, zip_codes: Iterable[str]) -> pd.DataFrame:
    """Most recent market values per ZIP, for scoring a property 'today'."""
    rows = []
    for zip_code in sorted({str(z).zfill(5) for z in zip_codes}):
        row = {"zip_code": zip_code}
        if market.zhvi is not None:
            series = market.zhvi.loc[market.zhvi["zip_code"] == zip_code].set_index("month_index")["zhvi"]
            if not series.empty:
                last = series.index.max()
                row["zhvi"] = float(series.loc[last])
                row["zhvi_yoy"] = float(series.loc[last] / series.get(last - 12, np.nan) - 1)
                row["zhvi_month_index"] = int(last)
        if market.inventory is not None:
            subset = market.inventory.loc[market.inventory["zip_code"] == zip_code]
            if not subset.empty:
                last = subset.loc[subset["month_index"].idxmax()]
                row.update({column: float(last[column]) for column in INVENTORY_FEATURES})
        if market.rates is not None and not market.rates.empty:
            last = market.rates.loc[market.rates["month_index"].idxmax()]
            row.update({column: float(last[column]) for column in RATE_FEATURES})
        if market.acs is not None:
            match = market.acs.loc[market.acs["zip_code"] == zip_code]
            if not match.empty:
                row.update({column: float(match.iloc[0][column]) for column in ACS_FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)
