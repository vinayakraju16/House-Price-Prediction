# Data workspace

## King County Seattle sale-price model

Place `Real Property Sales.zip`, `Residential Building.zip`, and `Parcel.zip` under
`data/raw/king-county/`. Run `python -m ml.data.king_county` from the repository root.
The preparation code reads only approved modeling columns; owner names, street addresses,
recording numbers, and other direct identifiers are excluded at read time. Raw and processed
data are ignored by Git. Filter counts and hashes are recorded in
`reports/king_county_data_audit.json` and the registered model metadata.

## Wake County, NC sale-price data

Download from Wake County Tax Administration
(<https://services.wake.gov/realdata_extracts/>) into `data/raw/wake-county/`:

- `Qualified_Sales_Past_24Months.xlsx` (appraiser-reviewed arm's-length sales, updated daily)
- `RealEstDataMMDDYYYY.zip` (full fixed-width real estate file; the newest dated file is used)

Run `python -m ml.data.wake_county` from the repository root (needs `openpyxl`). Owner names,
mailing addresses, and street addresses are skipped at read time. Output goes to
`data/processed/wake-county/sales.csv`; filter counts go to `reports/wake_county_data_audit.json`.
Wake County publishes no bedroom count, so this market's model has no `beds` feature.

## Harris County data workspace

This folder is reserved for the Harris Central Appraisal District (HCAD) public-data import.

1. Visit HCAD's public-data page and download `Real_acct_owner.zip` and `Real_building_land.zip` for the desired year, or use the downloader script if you have a direct base URL.
2. Place the ZIP files in a separate folder for that release, for example `data/raw/harris-county/2025/`.
3. Run either:

   ```powershell
   python ml/download_harris_county.py --year 2025 --base-url "https://example.org/hcad/2025"
   ```

   or, if the official direct URLs are known:

   ```powershell
   python ml/download_harris_county.py --year 2025 --account-url "https://.../Real_acct_owner.zip" --building-url "https://.../Real_building_land.zip"
   ```

   Then run:

   ```powershell
   python ml/ingest_harris_county.py --year 2025
   ```

The script creates `data/processed/harris-county-2026-residential.csv`, a normalized property-attribute table. HCAD values are appraisal/market-value records, not verified sale prices; do not label a model trained on this data as a sale-price predictor.

The existing 2026 downloads may stay directly in `data/raw/harris-county/` for compatibility, but use year folders for every new release. Raw downloads are ignored by Git because they are large and updated by HCAD. See `ml/HARRIS_COUNTY.md` for details.
