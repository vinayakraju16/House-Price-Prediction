# Harris County / Houston model plan

## Source and limitation

Use the official HCAD Public Data files:

- `Real_acct_owner.zip` — account, address, and value records.
- `Real_building_land.zip` — residential building and property attributes.

HCAD publishes appraisal database values. These are useful for a Harris County appraisal-value prototype, but they are not a substitute for verified MLS sale prices. The app must continue to label results as **experimental appraisal-value estimates** until a licensed comparable-sales source is available.

## Years and model validity

The downloaded release is a **2026 snapshot**. It can train a 2026 appraisal-value model, but it cannot by itself prove how well the model predicts changes over time. `real_acct.txt` includes prior-value fields, but they are not a replacement for full historical snapshots or verified sale prices.

Build the dataset by year:

```text
data/processed/harris-county-2024-residential.csv
data/processed/harris-county-2025-residential.csv
data/processed/harris-county-2026-residential.csv
```

When earlier HCAD releases are available, process each separately, add an appraisal year feature, train on earlier years, and evaluate on the newest year. For a true sale-price model, add a licensed MLS/comparable-sales source with sale date and price.

## Import

Download the two files from <https://hcad.org/hcad-online-services/pdata/>, place them under a folder for that release, then run:

```powershell
python ml/ingest_harris_county.py --year 2026
```

For historical releases, keep the identical filenames in separate folders:

```text
data/raw/harris-county/2024/Real_acct_owner.zip
data/raw/harris-county/2024/Real_building_land.zip
data/raw/harris-county/2025/Real_acct_owner.zip
data/raw/harris-county/2025/Real_building_land.zip
```

## Next data task

Train the Harris County baseline with 2024–2025 data and evaluate against 2026:

```powershell
python ml/train_harris_county.py
```

The default samples the large files for a repeatable baseline. After reviewing metrics, use `--full` only if the machine has enough memory. The output model and metadata are saved separately from the historical demo model.
