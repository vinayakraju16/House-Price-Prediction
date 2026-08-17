"""Prepare the raw Seattle datasets for training and legacy-holdout evaluation.

The cleaning rules are deterministic and intentionally retain valid luxury outliers.

Run from the repository root:
    python -m ml.data.prepare
"""

import json
from pathlib import Path

import pandas as pd

from ml.data.validation import validate_seattle_dataframe

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "Seattle" / "train.csv"
TEST_PATH = ROOT / "data" / "Seattle" / "test.csv"
OUTPUT_DIR = ROOT / "data" / "Seattle"
REPORT_PATH = ROOT / "reports" / "data_audit.json"

FEATURES = ["beds", "baths", "size", "lot_size", "zip_code"]
TARGET = "price"


def load_and_inspect(csv_path):
    """Load and return basic statistics."""
    df = pd.read_csv(csv_path)
    validation = validate_seattle_dataframe(df, stage="raw")
    statistics = {
        "rows": len(df),
        "cols": len(df.columns),
        "duplicates": df.duplicated().sum(),
        "duplicate_full_rows": df.duplicated(subset=FEATURES + [TARGET]).sum(),
        "missing_by_column": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return statistics, validation


def remove_duplicates(df):
    """Remove exact row duplicates."""
    before = len(df)
    df_clean = df.drop_duplicates()
    removed = before - len(df_clean)
    return df_clean, removed


def standardize_units(df):
    """Standardize size_units and lot_size_units to square feet."""
    df_std = df.copy()

    if "size_units" in df_std.columns:
        size_mask = df_std["size_units"].astype(str).str.lower().eq("acre")
        if size_mask.any():
            df_std.loc[size_mask, "size"] = df_std.loc[size_mask, "size"] * 43560
            print(f"  Converted {size_mask.sum()} size values from acres to sq ft")

    if "lot_size_units" in df_std.columns:
        lot_mask = df_std["lot_size_units"].astype(str).str.lower().eq("acre")
        if lot_mask.any():
            df_std.loc[lot_mask, "lot_size"] = df_std.loc[lot_mask, "lot_size"] * 43560
            print(f"  Converted {lot_mask.sum()} lot_size values from acres to sq ft")

    return df_std


def handle_missing_values(df):
    """Fill missing values with sensible defaults."""
    df_filled = df.copy()

    # lot_size: fill with 0 (not applicable)
    if "lot_size" in df_filled.columns:
        missing_lot = df_filled["lot_size"].isna().sum()
        df_filled["lot_size"] = df_filled["lot_size"].fillna(0)
        print(f"  Filled {missing_lot} missing lot_size values with 0")

    # numeric features: fill with median
    for col in ["beds", "baths", "size"]:
        if col in df_filled.columns and df_filled[col].isna().any():
            missing = df_filled[col].isna().sum()
            median_val = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(median_val)
            print(f"  Filled {missing} missing {col} values with median {median_val}")

    # zip_code: fill with mode
    if "zip_code" in df_filled.columns and df_filled["zip_code"].isna().any():
        missing = df_filled["zip_code"].isna().sum()
        mode_val = df_filled["zip_code"].mode()[0]
        df_filled["zip_code"] = df_filled["zip_code"].fillna(mode_val)
        print(f"  Filled {missing} missing zip_code values with mode {mode_val}")

    return df_filled


def audit_outliers(df):
    """Identify and analyze outliers in key columns."""
    outliers = {}

    # Price outliers (extreme values)
    q1_price, q3_price = df[TARGET].quantile([0.25, 0.75])
    iqr_price = q3_price - q1_price
    lower_bound = q1_price - 3 * iqr_price  # more conservative than 1.5 * IQR
    upper_bound = q3_price + 3 * iqr_price
    price_outliers = df[(df[TARGET] < lower_bound) | (df[TARGET] > upper_bound)]

    outliers["price"] = {
        "q1": float(q1_price),
        "q3": float(q3_price),
        "iqr": float(iqr_price),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": len(price_outliers),
        "outlier_rows": price_outliers.index.tolist(),
    }

    # Size outliers
    q1_size, q3_size = df["size"].quantile([0.25, 0.75])
    iqr_size = q3_size - q1_size
    size_outliers = df[
        (df["size"] < q1_size - 3 * iqr_size) | (df["size"] > q3_size + 3 * iqr_size)
    ]
    outliers["size"] = {
        "outlier_count": len(size_outliers),
        "min": float(df["size"].min()),
        "max": float(df["size"].max()),
        "mean": float(df["size"].mean()),
    }

    # Beds/baths outliers
    max_beds = df["beds"].max()
    max_baths = df["baths"].max()
    outliers["beds"] = {"max": float(max_beds), "count_gt_10": int((df["beds"] > 10).sum())}
    outliers["baths"] = {"max": float(max_baths), "count_gt_8": int((df["baths"] > 8).sum())}

    return outliers


def clean_dataset(csv_path, set_name="train"):
    """Execute full cleaning pipeline on a dataset."""
    print(f"\n--- Cleaning {set_name} dataset ---")

    # Load and inspect
    raw_stats, validation = load_and_inspect(csv_path)
    print(f"  Rows: {raw_stats['rows']}")
    print(f"  Duplicate full rows: {raw_stats['duplicate_full_rows']}")
    print(f"  Missing values: {raw_stats['missing_by_column']}")

    df = pd.read_csv(csv_path)

    # Remove duplicates
    df, dup_count = remove_duplicates(df)
    print(f"  Removed {dup_count} duplicate rows")

    # Standardize units
    df = standardize_units(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Validate and convert types
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    df["baths"] = pd.to_numeric(df["baths"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["lot_size"] = pd.to_numeric(df["lot_size"], errors="coerce")
    df["zip_code"] = pd.to_numeric(df["zip_code"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Drop any rows with NaN in target or key features
    before_dropna = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    dropped = before_dropna - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing key features")

    # Audit outliers
    outliers = audit_outliers(df)

    print("\nOutlier audit:")
    print(f"  Price outliers: {outliers['price']['outlier_count']}")
    print(f"  Size extreme values: min={outliers['size']['min']}, max={outliers['size']['max']}")
    print(f"  Beds: max={outliers['beds']['max']}, >10 beds: {outliers['beds']['count_gt_10']}")
    print(f"  Baths: max={outliers['baths']['max']}, >8 baths: {outliers['baths']['count_gt_8']}")

    # Select only the features we need
    df_final = df[FEATURES + [TARGET]].copy()

    # Final statistics
    print(f"\nFinal {set_name} dataset: {len(df_final)} rows")
    print(f"  Price: min={df_final[TARGET].min()}, median={df_final[TARGET].median()}, max={df_final[TARGET].max()}")
    print("  Feature ranges:")
    for col in FEATURES:
        print(f"    {col}: {df_final[col].min()} - {df_final[col].max()}")

    cleaned_validation = validate_seattle_dataframe(df_final, stage="cleaned")
    return df_final, raw_stats, outliers, validation, cleaned_validation


def main():
    """Prepare both raw Seattle datasets and write their validation audit."""
    print("=" * 60)
    print("SEATTLE DATA PREPARATION AND CLEANING")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Clean training data
    train_clean, train_raw, train_outliers, train_validation, train_cleaned_validation = clean_dataset(TRAIN_PATH, "train")
    train_output_path = OUTPUT_DIR / "train_cleaned.csv"
    train_clean.to_csv(train_output_path, index=False)
    print(f"  Saved cleaned training data to {train_output_path.name}")

    # Clean test data if it exists
    test_clean = None
    test_outliers = None
    if TEST_PATH.exists():
        test_clean, test_raw, test_outliers, test_validation, test_cleaned_validation = clean_dataset(TEST_PATH, "test")
        test_output_path = OUTPUT_DIR / "test_cleaned.csv"
        test_clean.to_csv(test_output_path, index=False)
        print(f"  Saved cleaned test data to {test_output_path.name}")

    # Generate comprehensive report
    report = {
        "pipeline": "Seattle data preparation",
        "dataset": "Seattle residential listings",
        "timestamp": pd.Timestamp.now().isoformat(),
        "train": {
            "raw": train_raw,
            "cleaned_rows": len(train_clean),
            "outliers": train_outliers,
            "raw_validation": train_validation,
            "cleaned_validation": train_cleaned_validation,
        },
    }

    if test_clean is not None:
        report["test"] = {
            "raw": test_raw,
            "cleaned_rows": len(test_clean),
            "outliers": test_outliers,
            "raw_validation": test_validation,
            "cleaned_validation": test_cleaned_validation,
        }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nData audit report saved to {REPORT_PATH.name}")
    print("\n" + "=" * 60)
    print("Data preparation complete. Ready for benchmarking.")
    print("=" * 60)


if __name__ == "__main__":
    main()
