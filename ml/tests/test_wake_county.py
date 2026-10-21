import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from ml.data.wake_county import (
    FULL_FILE_FIELDS,
    FULL_FILE_LINE_LENGTH,
    MODEL_FEATURES,
    _read_full_file,
    _sale_dates,
    find_full_file,
    grade_score,
    prepare_frames,
)


def _sale(reid, price=500000, serial=46000, book="020000", page=None, **overrides):
    page = page or reid[-5:]
    row = {
        "REID": reid, "SALE_DEED_BOOK": book, "SALE_DEED_PAGE": page, "SALE_PRICE": price,
        "SALE_DATE": serial, "SALE_TYPE": "Package",
        "PARCEL_DEEDED_ACREAGE_ATSALE": 0.25, "TOTAL_HEATED_AREA_ATSALE": 2000,
        "PARCEL_DEEDED_ACREAGE_CURRENT": 0.25, "TOTAL_HEATED_AREA_CURRENT": 2000,
        "BLDG_USE_DESC": "SINGLFAM", "YEAR_BUILT": 1995, "DESIGN_STYLE_DESC": "Conventional",
        "ZIP": "27615",
    }
    row.update(overrides)
    return row


def _card(reid, **overrides):
    row = {
        "REID": reid, "CARD": "01", "NUM_CARDS": "01", "YEAR_BUILT": "1995", "UNITS": "0",
        "HEATED_AREA": "2000", "YEAR_OF_ADDITION": "0", "EFFECTIVE_YEAR": "2000",
        "REMODELED_YEAR": "2015", "STORY_HEIGHT": "C", "DESIGN_STYLE": "A",
        "FOUNDATION_BASEMENT": "D", "EXTERIOR_WALL": "Z", "HEAT": "K", "AIR": "A",
        "BATH": "D", "BATH_FIXTURES": "0", "GRADE": "C+10", "CONDITION_PCT": "85",
        "TYPE_AND_USE": "01", "PHYSICAL_ZIP": "27615",
    }
    row.update(overrides)
    return row


class WakeCountyPreparationTests(unittest.TestCase):
    def test_filters_and_never_outputs_pii(self):
        sales = pd.DataFrame([
            _sale("0000001"),
            _sale("0000002", SALE_TYPE="Land"),
            _sale("0000003", book="030000", page="00001"),  # multi-parcel deed
            _sale("0000004", book="030000", page="00001"),
            _sale("0000005", TOTAL_HEATED_AREA_CURRENT=2600),  # changed since sale
            _sale("0000006", BLDG_USE_DESC="MOBILE"),
            _sale("0000007"),  # second card on parcel
        ])
        buildings = pd.DataFrame([
            _card("0000001"), _card("0000002"), _card("0000003"), _card("0000004"),
            _card("0000005"), _card("0000006"),
            _card("0000007", NUM_CARDS="02"), _card("0000007", CARD="02", NUM_CARDS="02"),
        ])

        result, audit = prepare_frames(sales, buildings)

        self.assertEqual(result["parcel_id"].tolist(), ["0000001"])
        row = result.iloc[0]
        self.assertEqual(row["sale_date"], "2025-12-09")
        self.assertEqual(row["baths"], 2.5)
        self.assertEqual(row["stories"], 2.0)
        self.assertAlmostEqual(row["grade"], 3.10)
        self.assertEqual(row["lot_size"], 10890)
        self.assertEqual(row["year_renovated"], 2015)
        self.assertEqual(row["has_central_air"], 1)
        self.assertEqual(row["zip_code"], "27615")
        self.assertTrue(set(MODEL_FEATURES).issubset(result.columns))
        self.assertNotIn("beds", result.columns)
        self.assertFalse({"STREET_NAME", "STREET_NUM", "Owner 1"}.intersection(result.columns))
        self.assertEqual(audit["filter_counts"]["output_rows"], 1)

    def test_sale_dates_accept_excel_serials_and_parsed_datetimes(self):
        serials = pd.Series([46000, 45566])
        parsed = pd.Series(pd.to_datetime(["2025-12-09", "2024-10-01"]))
        self.assertEqual(_sale_dates(serials).tolist(), parsed.tolist())
        self.assertEqual(_sale_dates(parsed).tolist(), parsed.tolist())

    def test_grade_score(self):
        self.assertEqual(grade_score("C"), 3.0)
        self.assertAlmostEqual(grade_score("B-05"), 3.95)
        self.assertAlmostEqual(grade_score("AA+10"), 6.10)
        self.assertTrue(pd.isna(grade_score("29.61")))
        self.assertTrue(pd.isna(grade_score(None)))

    def test_reads_fixed_width_zip_without_identifier_columns(self):
        line = [" "] * FULL_FILE_LINE_LENGTH
        line[0:10] = "OWNER NAME"
        values = _card("0000042")
        for name, start, width in FULL_FILE_FIELDS:
            line[start - 1:start - 1 + width] = str(values[name]).ljust(width)[:width]
        record = "".join(line)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RealEstData10132026.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("RealEstData10132026.txt", record + "\r\n" + record + "\r\n")
            (Path(tmp) / "RealEstData01012026.zip").write_bytes(path.read_bytes())
            self.assertEqual(find_full_file(Path(tmp)).name, "RealEstData10132026.zip")
            frame = _read_full_file(path)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.loc[0, "REID"], "0000042")
        self.assertEqual(frame.loc[0, "GRADE"], "C+10")
        self.assertEqual(frame.loc[0, "PHYSICAL_ZIP"], "27615")
        self.assertEqual(list(frame.columns), [name for name, _, _ in FULL_FILE_FIELDS])


if __name__ == "__main__":
    unittest.main()
