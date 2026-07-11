"""Build a non-PII Harris County 2026 residential appraisal dataset.

This processes the large official HCAD files in chunks. It intentionally keeps
the 2026 appraisal target separate from the historical Ames sale-price model.
"""

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_BASE_DIR = ROOT / "data" / "raw" / "harris-county"
CHUNK_SIZE = 100_000

BUILDING_COLUMNS = [
    "acct", "property_use_cd", "impr_tp", "impr_mdl_cd", "structure",
    "qa_cd", "date_erected", "eff", "yr_remodel", "im_sq_ft", "act_ar",
    "heat_ar", "gross_ar", "base_ar", "accrued_depr_pct",
]
ACCOUNT_COLUMNS = [
    "acct", "yr", "state_class", "school_dist", "Neighborhood_Code",
    "Neighborhood_Grp", "Market_Area_1", "Market_Area_2", "econ_area",
    "econ_bld_class", "yr_impr", "bld_ar", "land_ar", "acreage",
    "tot_mkt_val", "prior_tot_mkt_val", "new_construction_val", "certified_date",
]
NUMERIC_BUILDING_COLUMNS = [
    "date_erected", "eff", "yr_remodel", "im_sq_ft", "act_ar", "heat_ar",
    "gross_ar", "base_ar", "accrued_depr_pct",
]
NUMERIC_ACCOUNT_COLUMNS = [
    "yr", "yr_impr", "bld_ar", "land_ar", "acreage", "tot_mkt_val",
    "prior_tot_mkt_val", "new_construction_val",
]


def find_member(zip_path, member_name):
    with ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.lower().endswith(member_name.lower())]
    if not matches:
        raise ValueError(f"{member_name} was not found in {zip_path.name}.")
    return matches[0]


def raw_dir_for_year(year):
    year_directory = RAW_BASE_DIR / str(year)
    if year_directory.exists():
        return year_directory
    # Preserve compatibility with the already-downloaded 2026 files.
    if year == 2026:
        return RAW_BASE_DIR
    raise FileNotFoundError(f"Missing {year_directory}. Create it and place that year's HCAD ZIP files there.")


def require_file(raw_dir, name):
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Download it from HCAD Public Data first.")
    return path


def read_chunks(zip_path, member_name, columns):
    member = find_member(zip_path, member_name)
    with ZipFile(zip_path) as archive, archive.open(member) as source:
        yield from pd.read_csv(
            source, sep="\t", usecols=lambda column: column in columns,
            dtype=str, chunksize=CHUNK_SIZE, low_memory=False, encoding="cp1252",
        )


def build_building_summary(building_zip):
    summaries = []
    for chunk in read_chunks(building_zip, "building_res.txt", BUILDING_COLUMNS):
        for column in NUMERIC_BUILDING_COLUMNS:
            if column in chunk:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        aggregate = {column: "sum" for column in NUMERIC_BUILDING_COLUMNS if column in chunk}
        aggregate.update({column: "first" for column in ["property_use_cd", "impr_tp", "impr_mdl_cd", "structure", "qa_cd"] if column in chunk})
        summaries.append(chunk.groupby("acct", as_index=False).agg(aggregate))

    combined = pd.concat(summaries, ignore_index=True)
    aggregate = {column: "sum" for column in NUMERIC_BUILDING_COLUMNS if column in combined}
    aggregate.update({column: "first" for column in ["property_use_cd", "impr_tp", "impr_mdl_cd", "structure", "qa_cd"] if column in combined})
    return combined.groupby("acct", as_index=False).agg(aggregate)


def main(year):
    raw_dir = raw_dir_for_year(year)
    account_zip = require_file(raw_dir, "Real_acct_owner.zip")
    building_zip = require_file(raw_dir, "Real_building_land.zip")
    buildings = build_building_summary(building_zip)
    output_path = ROOT / "data" / "processed" / f"harris-county-{year}-residential.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    wrote_header = False
    kept_rows = 0

    for accounts in read_chunks(account_zip, "real_acct.txt", ACCOUNT_COLUMNS):
        for column in NUMERIC_ACCOUNT_COLUMNS:
            if column in accounts:
                accounts[column] = pd.to_numeric(accounts[column], errors="coerce")
        result = accounts.merge(buildings, on="acct", how="inner")
        # Keep records with a 2026 market value. No owner/mailing address fields are included.
        result = result[(result["yr"] == year) & (result["tot_mkt_val"] > 0)].copy()
        result["source"] = "HCAD public data"
        result["geography"] = "Harris County, Texas"
        result["target_type"] = f"{year} HCAD total market value"
        result.to_csv(output_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        kept_rows += len(result)

    print(f"Wrote {kept_rows:,} {year} residential appraisal records to {output_path}")
    print("Target: tot_mkt_val (HCAD appraisal market value, not a verified sale price).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026, help="Appraisal year in the downloaded HCAD release")
    main(parser.parse_args().year)
