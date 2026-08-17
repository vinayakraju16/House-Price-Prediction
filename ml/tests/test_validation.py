import unittest

import pandas as pd

from ml.data.validation import (
    DataValidationError,
    audit_seattle_dataframe,
    validate_seattle_dataframe,
)

VALID_CLEAN_ROW = {
    "beds": 3,
    "baths": 2.5,
    "size": 1800,
    "lot_size": 5000,
    "zip_code": 98144,
    "price": 850000,
}


class SeattleValidationTests(unittest.TestCase):
    def test_accepts_valid_cleaned_data(self):
        report = validate_seattle_dataframe(pd.DataFrame([VALID_CLEAN_ROW]), stage="cleaned")
        self.assertTrue(report["valid"])

    def test_rejects_nonpositive_size(self):
        frame = pd.DataFrame([{**VALID_CLEAN_ROW, "size": 0}])
        with self.assertRaisesRegex(DataValidationError, "size must be greater than zero"):
            validate_seattle_dataframe(frame, stage="cleaned")

    def test_rejects_invalid_zip(self):
        frame = pd.DataFrame([{**VALID_CLEAN_ROW, "zip_code": 9814}])
        with self.assertRaisesRegex(DataValidationError, "5-digit ZIP"):
            validate_seattle_dataframe(frame, stage="cleaned")

    def test_rejects_cleaned_duplicates(self):
        frame = pd.DataFrame([VALID_CLEAN_ROW, VALID_CLEAN_ROW])
        with self.assertRaisesRegex(DataValidationError, "duplicate"):
            validate_seattle_dataframe(frame, stage="cleaned")

    def test_raw_missing_lot_pair_is_allowed(self):
        raw = {
            "beds": 3, "baths": 2.5, "size": 1800, "size_units": "sqft",
            "lot_size": None, "lot_size_units": None, "zip_code": 98144, "price": 850000,
        }
        report = validate_seattle_dataframe(pd.DataFrame([raw]), stage="raw")
        self.assertTrue(report["valid"])

    def test_reports_target_outliers_without_deleting_them(self):
        rows = [{**VALID_CLEAN_ROW, "price": 500000 + index * 1000} for index in range(20)]
        rows.append({**VALID_CLEAN_ROW, "zip_code": 98145, "price": 25000000})
        report = audit_seattle_dataframe(pd.DataFrame(rows), stage="cleaned")
        self.assertGreater(report["target"]["iqr_outliers_3x"], 0)
