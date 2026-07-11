"""Train and evaluate a Harris County appraisal-value model.

Default mode samples the large files for a fast, repeatable baseline:
    python ml/train_harris_county.py

Use --full only after validating memory and training time on the machine.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "backend" / "mlmodels" / "harris_county_2026_model.joblib"
METADATA_PATH = ROOT / "backend" / "mlmodels" / "harris_county_2026_metadata.json"
TARGET = "tot_mkt_val"

CATEGORICAL_FEATURES = [
    "state_class", "school_dist", "Neighborhood_Code", "Neighborhood_Grp",
    "Market_Area_1", "Market_Area_2", "econ_area", "econ_bld_class",
    "property_use_cd", "impr_tp", "impr_mdl_cd", "structure", "qa_cd",
]
NUMERIC_FEATURES = [
    "yr_impr", "bld_ar", "land_ar", "acreage", "date_erected", "eff",
    "yr_remodel", "im_sq_ft", "act_ar", "heat_ar", "gross_ar", "base_ar",
    "accrued_depr_pct",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def read_sample(path, rows, seed):
    """Read a deterministic representative sample without loading the whole CSV."""
    if rows is None:
        return pd.read_csv(path, usecols=FEATURES + [TARGET])

    file_size = path.stat().st_size
    # The source files are about 250–320 MB and contain roughly 1.25M rows.
    estimated_rows = max(int(file_size / 250), rows)
    fraction = min(1.0, rows / estimated_rows)
    samples = []
    for index, chunk in enumerate(pd.read_csv(
        path, usecols=FEATURES + [TARGET], chunksize=100_000,
        dtype={column: "string" for column in CATEGORICAL_FEATURES},
    )):
        take = min(len(chunk), max(1, round(len(chunk) * fraction)))
        samples.append(chunk.sample(n=take, random_state=seed + index))
    combined = pd.concat(samples, ignore_index=True)
    return combined.sample(n=min(rows, len(combined)), random_state=seed).reset_index(drop=True)


def clean(data):
    data = data.copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("string").fillna("__missing__")
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[data[TARGET] > 0]
    # Remove implausible ultra-high values from the baseline; the threshold is
    # recorded in metadata and can be revisited after domain review.
    return data[data[TARGET] <= 5_000_000]


def main(train_rows, test_rows):
    train_paths = [
        DATA_DIR / "harris-county-2024-residential.csv",
        DATA_DIR / "harris-county-2025-residential.csv",
    ]
    test_path = DATA_DIR / "harris-county-2026-residential.csv"
    for path in [*train_paths, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing processed data file: {path}")

    train = pd.concat([clean(read_sample(path, train_rows // 2 if train_rows else None, 42 + index)) for index, path in enumerate(train_paths)], ignore_index=True)
    test = clean(read_sample(test_path, test_rows, 100))
    x_train, y_train = train[FEATURES], np.log1p(train[TARGET])
    x_test, y_test = test[FEATURES], test[TARGET]

    preprocess = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL_FEATURES),
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", HistGradientBoostingRegressor(
            learning_rate=0.08, max_iter=200, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, random_state=42,
        )),
    ])
    model.fit(x_train, y_train)
    predicted = np.expm1(model.predict(x_test))
    metadata = {
        "market": "Harris County, Texas",
        "appraisal_year": 2026,
        "target": "HCAD total market value (tot_mkt_val)",
        "target_disclaimer": "HCAD appraisal market value, not verified sale price.",
        "training_years": [2024, 2025],
        "test_year": 2026,
        "training_rows": len(train),
        "test_rows": len(test),
        "excluded_target_values_over": 5_000_000,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "example_input": {
            feature: _json_value(x_test.iloc[0][feature]) for feature in FEATURES
        },
        "metrics": {
            "mae": round(float(mean_absolute_error(y_test, predicted)), 2),
            "rmse": round(float(mean_squared_error(y_test, predicted) ** 0.5), 2),
            "r2": round(float(r2_score(y_test, predicted)), 4),
            "median_absolute_percentage_error": round(float(np.median(np.abs(predicted - y_test) / y_test) * 100), 2),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def _json_value(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=250_000)
    parser.add_argument("--test-rows", type=int, default=100_000)
    parser.add_argument("--full", action="store_true", help="Use every processed record; requires substantial memory.")
    args = parser.parse_args()
    main(None if args.full else args.train_rows, None if args.full else args.test_rows)
