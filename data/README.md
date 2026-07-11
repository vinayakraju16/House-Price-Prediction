# Harris County data workspace

This folder is reserved for the Harris Central Appraisal District (HCAD) public-data import.

1. Visit HCAD's public-data page and download `Real_acct_owner.zip` and `Real_building_land.zip` for the desired year.
2. Place the ZIP files in a separate folder for that release, for example `data/raw/harris-county/2025/`.
3. Run:

   ```powershell
   python ml/ingest_harris_county.py --year 2025
   ```

The script creates `data/processed/harris-county-2026-residential.csv`, a normalized property-attribute table. HCAD values are appraisal/market-value records, not verified sale prices; do not label a model trained on this data as a sale-price predictor.

The existing 2026 downloads may stay directly in `data/raw/harris-county/` for compatibility, but use year folders for every new release. Raw downloads are ignored by Git because they are large and updated by HCAD. See `ml/HARRIS_COUNTY.md` for details.
