import unittest

import pandas as pd

from ml.data.king_county import MODEL_FEATURES, prepare_frames


class KingCountyPreparationTests(unittest.TestCase):
    def test_filters_non_arm_length_and_never_outputs_pii(self):
        sales = pd.DataFrame([
        {"Major": "1", "Minor": "2", "DocumentDate": "2025-01-02", "SalePrice": 800000,
         "PropertyType": 11, "PrincipalUse": 6, "SaleInstrument": 3, "SaleReason": 1,
         "PropertyClass": 8, "SaleWarning": ""},
        {"Major": "3", "Minor": "4", "DocumentDate": "2025-01-02", "SalePrice": 1,
         "PropertyType": 11, "PrincipalUse": 6, "SaleInstrument": 3, "SaleReason": 18,
         "PropertyClass": 8, "SaleWarning": ""},
    ])
        buildings = pd.DataFrame([
        {"Major": "1", "Minor": "2", "ZipCode": 98144, "Stories": 2, "BldgGrade": 8,
         "SqFtTotLiving": 2200, "SqFtFinBasement": 400, "SqFtGarageBasement": 0,
         "SqFtGarageAttached": 300, "Bedrooms": 3, "BathHalfCount": 1,
         "Bath3qtrCount": 0, "BathFullCount": 2, "FpSingleStory": 1,
         "FpMultiStory": 0, "FpFreestanding": 0, "FpAdditional": 0, "YrBuilt": 1990,
         "YrRenovated": 2018, "Condition": 3, "HeatSystem": 5, "ViewUtilization": "Y"},
    ])
        parcels = pd.DataFrame([{"Major": "1", "Minor": "2", "PresentUse": 2, "SqFtLot": 5000}])

        result, audit = prepare_frames(sales, buildings, parcels)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "baths"], 2.5)
        self.assertEqual(result.loc[0, "garage_sqft"], 300)
        self.assertTrue(set(MODEL_FEATURES).issubset(result.columns))
        self.assertFalse({"SellerName", "BuyerName", "Address"}.intersection(result.columns))
        self.assertEqual(audit["filter_counts"]["output_rows"], 1)


if __name__ == "__main__":
    unittest.main()
