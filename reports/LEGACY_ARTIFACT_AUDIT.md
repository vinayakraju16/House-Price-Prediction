# Legacy Artifact Audit and Cleanup Record

Audit and consolidation date: August 16, 2026
Method: reference search, schema comparison, file sizes, and SHA-256 hashing
Safety: reversible archive moves; no file was permanently deleted

## Outcome

The active repository now has one clear Seattle ML path:

```text
data/Seattle/{train,test}.csv
  -> ml.data.prepare
  -> data/Seattle/{train,test}_cleaned.csv
  -> ml.training.benchmark / tune / train_final
  -> backend/mlmodels/model_registry.json
  -> Django cached inference service
```

Thirty-five source entries containing 52 files and 9,069,270 bytes were moved out of active
paths and into `archive/`. These include unrelated datasets, superseded code and reports,
notebooks, unregistered model files, and retired registry versions.

Registry version `2.2.0` remains active. `seattle-ridge-v1` remains as a verified,
structurally different rollback. Active model and metadata hashes match the registry.

## Canonical paths

| Purpose | Canonical path |
|---|---|
| Raw Seattle data | `data/Seattle/train.csv` and `test.csv` |
| Reproducible cleaning | `ml/data/prepare.py` |
| Data validation | `ml/data/validation.py` |
| Feature construction | `ml/features/build_features.py` |
| Benchmarking and tuning | `ml/training/benchmark.py` and `tune.py` |
| Final training/registration | `ml/training/train_final.py` |
| Current evaluation | `reports/MODEL_EVALUATION.md` |
| Leakage investigation | `reports/PRODUCTION_MODEL_AUDIT.md` |
| Model selection | `backend/mlmodels/model_registry.json` |
| Active Seattle artifact | `backend/mlmodels/versions/2.2.0/` |
| Seattle rollback | `backend/mlmodels/versions/seattle-ridge-v1.*` |
| Separate Harris product | `backend/mlmodels/harris_county_2026_*` |

`backend/mlmodels/trained_models.pkl` and `metadata.json` are compatibility mirrors generated
by final training. They are not the registry source of truth, but the service intentionally
uses them if the registry is absent.

## Findings and actions

### Target-leaked Phase 1 evaluation

The old Phase 1 pipeline used `price_per_sqft` and `is_price_anomaly`, both derived from the
target sale price. Its reported high R2 and low MAPE were invalid production claims. The old
scripts, derived CSVs, summaries, JSON results, and completion report now live under
`archive/seattle-legacy/`. The production audit remains active as the corrective record.

### Exact model duplicates

Four model pairs were byte-for-byte duplicates:

| Artifact name | SHA-256 | Duplicate bytes |
|---|---|---:|
| `elastic_net_model.pkl` | `2f60bf138a998a86c857c2b3af0085fd60384d6a5b5c7671898de7def7acc4e5` | 21,818 |
| `gbr_feature_names.pkl` | `9e0ed7f31099a700bb84eb8738650ba44806e8f2efedf5189303ec3c8c14802f` | 733 |
| `gbr_model.pkl` | `e1d60566a1cb32d2a3eaf70d5e38360d05ebb9b030d9d51609ba2af21316e787` | 356,220 |
| `gradient_boost.pkl` | `0ba3c651e1f9ccfda54d09d2475e52469cd59ae9a1e99fed27a8e4397d391a8b` | 663,966 |

Both unregistered deployment-area copies were moved under `archive/model-artifacts/` along
with the non-identical pre-registry artifact and empty wheel. No archived pickle/joblib file
is loaded or deserialized by the application.

### Unrelated datasets

The former `GB_V2` experiment and root selected-feature CSV use the 1,460-row Ames Housing
schema, not the active Seattle schema. California Housing and Realtor country-history files
also had no active code references. They now live under `archive/datasets/` and must not be
mixed into Seattle evaluation.

HCAD raw/processed data was retained because the separate Harris appraisal endpoint and
training pipeline actively use it. It must not be presented as verified sale-price data.

### Reproducible data preparation

The useful logic from the old phase-named cleaner was promoted to `ml/data/prepare.py`.
Regression tests prove that it reproduces the checked-in 2,007-row cleaned training dataset
and 504-row cleaned legacy holdout from their raw inputs.

### Model retention

Experimental registry versions 2.0.0 and 2.1.0 were retired to the archive. The policy now
keeps the active checksummed artifact plus one diverse rollback. Registry entries are removed
before their files leave the serving directory, preventing dangling model references.

## Archive map

| Archive path | Contents |
|---|---|
| `archive/datasets/ames/` | Ames data, notebook, preprocessing, and model outputs |
| `archive/datasets/unrelated/` | California Housing and Realtor country history |
| `archive/notebooks/` | Superseded exploratory/model notebooks |
| `archive/seattle-legacy/` | Phase 0/1 scripts, reports, summaries, and derived data |
| `archive/model-artifacts/` | Unregistered, duplicate, and retired artifacts |
| `archive/preexisting-ml-archive/` | Earlier repository archive |

See `archive/README.md` for restoration and trust rules.

## Permanent-deletion policy

The archive is intentionally retained until its removal receives explicit, path-level
approval after a clean test run. Before permanent deletion:

1. confirm no active reference outside `archive/`;
2. verify registry hashes and readiness;
3. run ML, backend, frontend, build, and browser tests;
4. preserve any provenance the project owner still wants;
5. delete only explicit resolved paths inside `archive/`, never a computed broad target.

The repository also contains pre-existing working-tree deletions for formerly vendored Python
packages, generated frontend files, and obsolete `FinalCode` material. Those changes were not
created by this cleanup and should be reviewed as a separate commit boundary.
