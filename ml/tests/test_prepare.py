"""Regression tests for the reproducible Seattle cleaning pipeline."""

from pathlib import Path
from unittest import TestCase

import pandas as pd

from ml.data.prepare import clean_dataset

ROOT = Path(__file__).resolve().parents[2]
SEATTLE_DATA = ROOT / "data" / "Seattle"


class SeattlePreparationTests(TestCase):
    def test_raw_training_data_reproduces_checked_in_cleaned_data(self):
        actual, *_ = clean_dataset(SEATTLE_DATA / "train.csv", "train")
        expected = pd.read_csv(SEATTLE_DATA / "train_cleaned.csv")

        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_raw_legacy_holdout_reproduces_checked_in_cleaned_data(self):
        actual, *_ = clean_dataset(SEATTLE_DATA / "test.csv", "legacy holdout")
        expected = pd.read_csv(SEATTLE_DATA / "test_cleaned.csv")

        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )
