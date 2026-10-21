import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.market_features import (
    ACS_FILE,
    INVENTORY_FILE,
    MARKET_FEATURES,
    MORTGAGE_FILE,
    ZHVI_FILE,
    MarketData,
    add_market_features,
    latest_snapshot,
)


def _write_sources(folder: Path) -> None:
    months = pd.date_range("2024-01-31", "2025-06-30", freq="ME").strftime("%Y-%m-%d")
    zhvi = {"RegionID": [1, 2], "SizeRank": [1, 2], "RegionName": ["27615", "8701"],
            "RegionType": "zip", "StateName": "NC", "State": "NC", "City": "x",
            "Metro": "x", "CountyName": "x"}
    for index, month in enumerate(months):
        zhvi[month] = [400_000 + 1_000 * index, 200_000.0]
    pd.DataFrame(zhvi).to_csv(folder / ZHVI_FILE, index=False)

    inventory = []
    for index, month in enumerate(pd.period_range("2024-01", "2025-06", freq="M")):
        if month == pd.Period("2025-03", freq="M"):
            continue  # a missing month must not shrink the smoothing window
        inventory.append({
            "month_date_yyyymm": month.year * 100 + month.month, "postal_code": 27615,
            "median_days_on_market": 30 + index, "price_reduced_share": 0.1,
            "active_listing_count_yy": 50.0, "pending_ratio": 0.5,
        })
    pd.DataFrame(inventory).to_csv(folder / INVENTORY_FILE, index=False)

    weeks = pd.date_range("2024-01-04", "2025-06-30", freq="7D")
    pd.DataFrame({
        "date": [f"{d.month}/{d.day}/{d.year}" for d in weeks],  # portable m/d/Y
        "pmms30": np.where(weeks.year == 2024, 7.0, 6.0),
    }).to_csv(folder / MORTGAGE_FILE, index=False)
    pd.DataFrame({
        "zcta": ["27615", "00601"], "median_household_income": [100_000, -666666666],
        "median_home_value": [450_000, 90_000], "median_gross_rent": [1500, 400],
        "occupied_units": [1000, 10], "owner_occupied_units": [700, 5], "population": [3, 4],
    }).to_csv(folder / ACS_FILE, index=False)


class MarketFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        _write_sources(self.folder)
        self.market = MarketData.load(self.folder)

    def tearDown(self):
        self.tmp.cleanup()

    def test_uses_previous_month_and_year_over_year(self):
        sales = pd.DataFrame({"sale_date": ["2025-05-20"], "zip_code": ["27615"]})
        result = add_market_features(sales, self.market).iloc[0]
        # May 2025 sale -> April 2025 data (index 15 from Jan 2024) and April 2024 (index 3).
        self.assertEqual(result["zhvi"], 415_000)
        self.assertAlmostEqual(result["zhvi_yoy"], 415_000 / 403_000 - 1)
        self.assertEqual(result["mortgage_rate_30y"], 6.0)
        self.assertEqual(result["mortgage_rate_change_1y"], -1.0)
        self.assertEqual(result["median_household_income"], 100_000)
        self.assertAlmostEqual(result["owner_occupied_share"], 0.7)
        self.assertEqual(result["inventory_yoy"], 3.0)  # 50x clipped to +300%

    def test_smoothing_spans_calendar_months_with_gaps(self):
        sales = pd.DataFrame({"sale_date": ["2025-05-01"], "zip_code": ["27615"]})
        result = add_market_features(sales, self.market).iloc[0]
        # April 2025 window = Feb (index 13, dom 43), Mar (missing), Apr (index 15, dom 45).
        self.assertAlmostEqual(result["days_on_market"], 44.0)

    def test_unknown_zip_and_early_dates_are_missing_not_wrong(self):
        sales = pd.DataFrame({"sale_date": ["2025-05-01", "2024-01-15"],
                              "zip_code": ["99999", "27615"]})
        result = add_market_features(sales, self.market)
        self.assertTrue(result.loc[0, ["zhvi", "days_on_market", "median_household_income"]]
                        .isna().all())
        self.assertTrue(np.isnan(result.loc[1, "zhvi"]))  # Dec 2023 is before the data
        self.assertTrue(set(MARKET_FEATURES).issubset(result.columns))

    def test_zero_padded_zip_and_suppressed_acs_values(self):
        sales = pd.DataFrame({"sale_date": ["2025-03-10"], "zip_code": ["08701"]})
        self.assertEqual(add_market_features(sales, self.market).loc[0, "zhvi"], 200_000)
        acs = self.market.acs.set_index("zip_code")
        self.assertTrue(np.isnan(acs.loc["00601", "median_household_income"]))

    def test_latest_snapshot(self):
        snapshot = latest_snapshot(self.market, ["27615"]).iloc[0]
        self.assertEqual(snapshot["zhvi"], 417_000)
        self.assertEqual(snapshot["mortgage_rate_30y"], 6.0)


if __name__ == "__main__":
    unittest.main()
