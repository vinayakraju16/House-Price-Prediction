"""Export a per-market model (property + ZIP market features) for the browser runtime.

    python -m ml.training.export_market_model                 # Wake County model + comparables
    python -m ml.training.export_market_model --fixtures      # also write parity fixtures

Writes ``frontend/src/model/<market>-<version>.json`` (format version 2, evaluated by
``frontend/src/model/marketPredictor.js``), ``<market>-comparables-<version>.json`` and
``markets.json``, the ZIP-to-market routing table the app loads first.

Format 2 differs from the King County format 1 in three ways: the feature engineer is the
configurable ``PropertyMarketFeatureEngineer``; the target can be ZHVI-relative
(price = exp(raw + log ZHVI)); and each model carries the latest market snapshot per ZIP
so the browser can fill market features for "today" without any request.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.data.market_features import MARKET_DIR, ZHVI_FILE
from ml.training.export_browser_model import _finite, _sha256, _tree_payload, _write_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from model_runtime.market import (  # noqa: E402
    MarketRelativeRegressor,
    PropertyMarketFeatureEngineer,
)

MODEL_DIR = ROOT / "backend" / "mlmodels" / "versions"
OUTPUT_DIR = ROOT / "frontend" / "src" / "model"
FIXTURE_DIR = OUTPUT_DIR / "__fixtures__"
FORMAT = "havenvalue-browser-model"
FORMAT_VERSION = 2
SHAPLEY_PERMUTATIONS = 128
SHAPLEY_SEED = 42
SNAPSHOT_FIELDS = (
    "zhvi", "zhvi_yoy", "days_on_market", "price_reduced_share", "inventory_yoy",
    "pending_ratio", "mortgage_rate_30y", "mortgage_rate_change_1y",
    "median_household_income", "owner_occupied_share",
)
MARKETS = {
    "wake-county": {
        "version": "wake-county-1.0.0",
        "name": "Wake County, NC",
        "short_name": "Wake County",
        "source": "Wake County qualified sale",
        "disclaimer": ("Experimental estimate based on Wake County Tax Administration qualified "
                       "sales and ZIP-level market data; not a formal appraisal."),
    },
}
KING_COUNTY_ROUTE = {
    "id": "king-county", "name": "Seattle (King County, WA)", "short_name": "Seattle",
    "zip_ranges": [[98101, 98199]],
}


def _unwrap(model):
    """Return (pipeline, target, relative-parameters) for a fitted market model."""
    if isinstance(model, MarketRelativeRegressor):
        relative = {
            "index_column": model.index_column,
            "zip_log_index": {str(k): float(v) for k, v in model.zip_log_index_.items()},
            "global_log_index": float(model.global_log_index_),
        }
        return model.regressor_, "zhvi_relative", relative
    if getattr(model, "func", None) is not None and model.func.__name__ == "log1p":
        return model.regressor_, "log1p", None
    raise RuntimeError(f"Unsupported model wrapper: {type(model).__name__}")


def build_market_model(market_id: str, bundle: dict, metadata: dict) -> dict:
    pipeline, target, relative = _unwrap(bundle["best_model"])
    preprocessor, regressor = (step for _, step in pipeline.steps)
    engineer, columns = (step for _, step in preprocessor.steps)
    if not isinstance(engineer, PropertyMarketFeatureEngineer):
        raise RuntimeError("Format 2 expects PropertyMarketFeatureEngineer.")
    transformers = {name: (pipe, list(cols)) for name, pipe, cols in columns.transformers_}
    numeric_pipeline, numeric_columns = transformers["numeric"]
    categorical_pipeline, categorical_columns = transformers["categorical"]
    encoder = categorical_pipeline.named_steps["onehot"]
    if encoder.handle_unknown != "ignore" or encoder.drop is not None:
        raise RuntimeError("Expected OneHotEncoder(handle_unknown='ignore').")

    features = list(bundle["feature_names"])
    rng = np.random.default_rng(SHAPLEY_SEED)
    permutations = [rng.permutation(len(features)).tolist() for _ in range(SHAPLEY_PERMUTATIONS)]
    snapshot = {
        row["zip_code"]: {field: _finite(row.get(field)) for field in SNAPSHOT_FIELDS}
        for row in bundle["market_snapshot"]
    }
    month_index = metadata.get("market_snapshot_month_index")
    reference = dict(metadata["feature_defaults"])
    reference.update(market_reference(snapshot, engineer.market))
    info = MARKETS[market_id]
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "market": market_id,
        "market_name": info["name"],
        "model_version": metadata["model_version"],
        "artifact_sha256": metadata["artifact"]["sha256"],
        "features": features,
        "inputs": {
            "numeric": list(engineer.numeric),
            "categorical": list(engineer.categorical),
            "market": list(engineer.market),
            "excluded": list(engineer.exclude),
            "required": metadata["required_features"],
            "categories": {
                column: [str(value) for value in values]
                for column, values in zip(categorical_columns, encoder.categories_, strict=True)
            },
        },
        "zip_codes": metadata["zip_codes"],
        "feature_defaults": {k: _finite(v) for k, v in metadata["feature_defaults"].items()},
        "typical_feature_ranges": metadata["typical_feature_ranges"],
        "explanation_reference": {k: _finite(v) for k, v in reference.items()},
        "prediction_interval": metadata["prediction_interval"],
        "market_snapshot": {
            "month": None if month_index is None else
            f"{month_index // 12:04d}-{month_index % 12 + 1:02d}",
            "zips": snapshot,
        },
        "model_info": {
            "model_version": metadata["model_version"], "status": metadata["status"],
            "market": info["name"], "training_rows": metadata["training_rows"],
            "training_date": metadata["training_date"], "metrics": metadata["metrics"],
            "target": metadata["target"], "variant": metadata["variant"],
        },
        "disclaimer": info["disclaimer"],
        "comparable_source": info["source"],
        "preprocessing": {
            "numeric": {
                "columns": numeric_columns,
                "impute": numeric_pipeline.named_steps["imputer"].statistics_.astype(float).tolist(),
                "mean": numeric_pipeline.named_steps["scaler"].mean_.astype(float).tolist(),
                "scale": numeric_pipeline.named_steps["scaler"].scale_.astype(float).tolist(),
            },
            "categorical": {
                "columns": categorical_columns,
                "impute": [str(v) for v in categorical_pipeline.named_steps["imputer"].statistics_],
                "categories": [[str(v) for v in values] for values in encoder.categories_],
            },
        },
        "target_transform": target,
        "relative": relative,
        "regressor": _tree_payload(regressor),
        "shapley": {"seed": SHAPLEY_SEED, "permutations": permutations},
    }


def market_reference(snapshot: dict, market_columns) -> dict:
    """Typical market values (median across ZIPs) used as the explanation baseline."""
    reference = {}
    for column in market_columns:
        values = [row[column] for row in snapshot.values() if row.get(column) is not None]
        reference[column] = float(np.median(values)) if values else None
    return reference


def build_comparables(bundle: dict) -> dict:
    # Records without the fields the distance uses cannot be ranked; leave them out.
    data = pd.DataFrame(bundle["comparables"]).dropna(subset=["baths", "size", "lot_size", "price"])
    columns = {
        "baths": [_finite(v) for v in data["baths"].astype(float)],
        "size": [_finite(v) for v in data["size"].astype(float)],
        "lot_size": [_finite(v) for v in data["lot_size"].astype(float)],
        "zip_code": data["zip_code"].astype(str).tolist(),
        "price": [_finite(v) for v in data["price"].astype(float)],
        "sale_date": data["sale_date"].astype(str).tolist(),
        "year_built": [None if pd.isna(v) else int(v) for v in data["year_built"]],
        "property_type": data["property_type"].astype(str).tolist(),
    }
    log_scales = {}
    for feature in ("size", "lot_size"):
        values = np.log1p(data[feature].to_numpy(dtype=float))
        log_scales[feature] = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    return {"rows": len(data), "columns": columns, "log_scales": log_scales}


# ---------------------------------------------------------------------------------------
# Python reference implementation of the browser behaviour, used for parity fixtures.


def build_row(browser: dict, payload: dict, year: int, month: int) -> dict:
    row = {name: browser["feature_defaults"].get(name) for name in browser["features"]}
    row.update(payload)
    row["sale_year"], row["sale_month"] = year, month
    market = browser["market_snapshot"]["zips"].get(str(payload["zip_code"]), {})
    for column in browser["inputs"]["market"]:
        row[column] = market.get(column)
    return row


def _frame(rows: list[dict], features: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=features)
    return frame.astype({name: object for name in features})


def explain(model, browser: dict, row: dict) -> dict:
    features = browser["features"]
    reference = browser["explanation_reference"]
    paths = []
    for order in browser["shapley"]["permutations"]:
        path = {name: reference[name] if name in reference else row[name] for name in features}
        paths.append(dict(path))
        for index in order:
            path[features[index]] = row[features[index]]
            paths.append(dict(path))
    predictions = model.predict(_frame(paths, features))
    step = len(features) + 1
    totals = np.zeros(len(features))
    base = []
    for p, order in enumerate(browser["shapley"]["permutations"]):
        chunk = predictions[p * step:(p + 1) * step]
        base.append(chunk[0])
        for position, index in enumerate(order):
            totals[index] += chunk[position + 1] - chunk[position]
    count = len(browser["shapley"]["permutations"])
    return {"base_value": float(np.mean(base)),
            "contributions": {name: float(totals[i] / count) for i, name in enumerate(features)}}


def comparables_for(browser: dict, comparables: dict, row: dict, limit: int = 5) -> list[dict]:
    columns = comparables["columns"]
    ranges = browser["typical_feature_ranges"]
    distance = np.zeros(comparables["rows"])
    bounds = ranges["baths"]
    distance += np.abs(np.asarray(columns["baths"], float) - float(row["baths"])) / max(
        bounds["p95"] - bounds["p05"], 1)
    for feature, weight in (("size", 2.0), ("lot_size", 0.5)):
        scale = max(comparables["log_scales"][feature], 1)
        values = np.log1p(np.asarray(columns[feature], float))
        distance += weight * np.abs(values - math.log1p(float(row[feature]))) / scale
    distance += np.where(np.asarray(columns["zip_code"]) != str(row["zip_code"]), 2.0, 0.0)
    order = np.lexsort((np.arange(len(distance)), distance))[:limit]
    return [{"price": columns["price"][i], "similarity": float(1 / (1 + distance[i]))}
            for i in order]


def _random_payloads(rng, browser: dict, count: int) -> list[dict]:
    categories = browser["inputs"]["categories"]
    zips = browser["zip_codes"]
    payloads = []
    for _ in range(count):
        payload = {
            "baths": float(rng.choice([1, 1.5, 2, 2.5, 3, 3.5])),
            "size": int(rng.integers(600, 6500)),
            "lot_size": int(rng.choice([0, 1500, 4000, 8000, 10890, 21780, 87120])),
            "zip_code": str(rng.choice(zips)),
        }
        optional = {
            "stories": float(rng.choice([1, 1.5, 2, 3])),
            "grade": float(rng.choice([2.0, 3.0, 3.1, 3.95, 4.0, 5.0, 6.0])),
            "condition": int(rng.integers(40, 120)),
            "year_built": int(rng.integers(1940, 2027)),
            "has_basement": int(rng.integers(0, 2)),
            "has_central_air": int(rng.integers(0, 2)),
            "heat_system": str(rng.choice(categories["heat_system"])),
            "exterior_wall": str(rng.choice(categories["exterior_wall"])),
            "property_type": str(rng.choice(categories["property_type"])),
        }
        for name, value in optional.items():
            if rng.random() < 0.6:
                payload[name] = value
        if "year_built" in payload and rng.random() < 0.3:
            payload["year_renovated"] = int(rng.integers(payload["year_built"], 2027))
        payloads.append(payload)
    return payloads


def build_fixtures(model, browser: dict, comparables: dict, count: int = 40,
                   year: int = 2026, month: int = 10) -> dict:
    rng = np.random.default_rng(2026)
    cases = []
    for payload in _random_payloads(rng, browser, count):
        row = build_row(browser, payload, year, month)
        prediction = float(model.predict(_frame([row], browser["features"]))[0])
        explanation = explain(model, browser, row)
        found = comparables_for(browser, comparables, row)
        cases.append({
            "payload": payload, "prediction": prediction,
            "base_value": explanation["base_value"],
            "contributions": explanation["contributions"],
            "comparable_prices": [item["price"] for item in found],
            "comparable_similarity": [item["similarity"] for item in found],
        })
    return {"sale_year": year, "sale_month": month, "cases": cases}


# ZIP directory shown as suggestions in the estimate form. ZIPs with fewer training sales
# than this are not suggested, and the form flags them as limited data.
MIN_ZIP_SALES = 30
PROCESSED_DIR = ROOT / "data" / "processed"
ZIP_PLACE_NAMES = {
    # Seattle neighbourhoods (the Zillow file only says "Seattle").
    "98102": "Capitol Hill / Eastlake", "98103": "Fremont / Wallingford / Greenwood",
    "98105": "University District / Laurelhurst", "98106": "Delridge / Highland Park",
    "98107": "Ballard", "98108": "Georgetown / Beacon Hill",
    "98109": "Queen Anne / South Lake Union", "98112": "Madison Park / Montlake",
    "98115": "Ravenna / Wedgwood", "98116": "West Seattle (Alki)",
    "98117": "Ballard / Crown Hill", "98118": "Columbia City / Rainier Valley",
    "98119": "Queen Anne / Interbay", "98121": "Belltown", "98122": "Central District / Madrona",
    "98125": "Lake City / Northgate", "98126": "West Seattle (High Point)",
    "98133": "Bitter Lake / Shoreline", "98134": "SoDo", "98136": "West Seattle (Fauntleroy)",
    "98144": "Mount Baker / Beacon Hill", "98146": "White Center / Burien",
    "98155": "Shoreline / Lake Forest Park", "98166": "Burien / Normandy Park",
    "98168": "Tukwila / Burien", "98177": "Broadview / Shoreline",
    "98178": "Skyway / Rainier Beach", "98188": "SeaTac / Tukwila",
    "98198": "Des Moines / SeaTac", "98199": "Magnolia",
    # Raleigh areas (the Zillow file only says "Raleigh").
    "27601": "Downtown", "27603": "South", "27604": "East", "27605": "Hillsborough Street",
    "27606": "Southwest", "27607": "West / NC State", "27608": "Five Points",
    "27609": "North Hills", "27610": "Southeast", "27612": "Northwest / Crabtree",
    "27613": "Northwest", "27614": "North / Wakefield", "27615": "North",
    "27616": "Northeast", "27617": "Brier Creek",
}
CITY_FIXES = {"Seatac": "SeaTac", "Fuquay Varina": "Fuquay-Varina"}
DIRECTORY_SOURCES = {
    "king-county": PROCESSED_DIR / "king-county" / "sales.csv",
    "wake-county": PROCESSED_DIR / "wake-county" / "sales.csv",
}


def zip_directory(market_id: str, zhvi_path: Path) -> list[dict]:
    """Per-ZIP training coverage (sales, median price), place name and latest ZHVI."""
    sales = pd.read_csv(DIRECTORY_SOURCES[market_id], usecols=["zip_code", "price"],
                        dtype={"zip_code": str})
    stats = sales.groupby("zip_code")["price"].agg(["size", "median"])
    zillow = pd.read_csv(zhvi_path, nrows=None, dtype={"RegionName": str},
                         usecols=lambda c: c in {"RegionName", "City"} or c[:4].isdigit())
    zillow["zip_code"] = zillow["RegionName"].str.zfill(5)
    zillow = zillow.set_index("zip_code")
    months = [c for c in zillow.columns if c[:4].isdigit()]
    city = market_id == "wake-county"
    entries = []
    for zip_code, row in stats.iterrows():
        place = zillow["City"].get(zip_code)
        place = None if pd.isna(place) else CITY_FIXES.get(str(place), str(place))
        area = ZIP_PLACE_NAMES.get(zip_code)
        if city and area and place:
            place = f"{place} ({area})"
        elif area:
            place = area
        latest = zillow.loc[zip_code, months].dropna().iloc[-1] if zip_code in zillow.index \
            and zillow.loc[zip_code, months].notna().any() else None
        entries.append({
            "zip": zip_code, "place": place, "sales": int(row["size"]),
            "median_price": float(row["median"]),
            "typical_value": None if latest is None else round(float(latest), -2),
            "well_covered": bool(row["size"] >= MIN_ZIP_SALES),
        })
    return entries


def write_routes(markets: dict[str, dict], zhvi_path: Path) -> None:
    routes = [{**KING_COUNTY_ROUTE, "zips": zip_directory("king-county", zhvi_path)}]
    for market_id, zip_codes in markets.items():
        routes.append({"id": market_id, "name": MARKETS[market_id]["name"],
                       "short_name": MARKETS[market_id]["short_name"],
                       "zip_codes": zip_codes, "zips": zip_directory(market_id, zhvi_path)})
    _write_json(OUTPUT_DIR / "markets.json", {"min_zip_sales": MIN_ZIP_SALES, "markets": routes})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--routes-only", action="store_true",
                        help="only rebuild markets.json (ZIP routing and ZIP directory)")
    parser.add_argument("--market-dir", type=Path, default=MARKET_DIR)
    args = parser.parse_args(argv)
    zhvi_path = args.market_dir / ZHVI_FILE
    if args.routes_only:
        zip_codes = {market_id: json.loads((MODEL_DIR / info["version"] / "metadata.json")
                                           .read_text(encoding="utf-8"))["zip_codes"]
                     for market_id, info in MARKETS.items()}
        write_routes(zip_codes, zhvi_path)
        return
    exported = {}
    for market_id, info in MARKETS.items():
        version_dir = MODEL_DIR / info["version"]
        metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
        model_path = version_dir / "model.joblib"
        if _sha256(model_path) != metadata["artifact"]["sha256"]:
            raise RuntimeError(f"{model_path} does not match its metadata checksum")
        bundle = joblib.load(model_path)
        browser = build_market_model(market_id, bundle, metadata)
        comparables = build_comparables(bundle)
        _write_json(OUTPUT_DIR / f"{info['version']}.json", browser)
        _write_json(OUTPUT_DIR / f"{market_id}-comparables-{metadata['model_version'].split('-')[-1]}.json",
                    comparables)
        if args.fixtures:
            _write_json(FIXTURE_DIR / f"parity-{info['version']}.json",
                        build_fixtures(bundle["best_model"], browser, comparables))
        exported[market_id] = browser["zip_codes"]
    write_routes(exported, zhvi_path)


if __name__ == "__main__":
    main()
