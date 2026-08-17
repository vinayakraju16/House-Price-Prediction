# Model artifacts

The Django Seattle endpoint resolves its active artifact through `model_registry.json`. Version
3.0.0 is trained from privacy-safe King County Assessor extracts and bundles the preprocessing
pipeline plus a compact comparable-sale reference set.
The compatibility copies `trained_models.pkl` and `metadata.json` mirror the active
version but are not the source of truth when the registry exists.

New registry entries include SHA-256 hashes for both artifact and metadata. Django verifies
those hashes before loading the corresponding file, and model-loading tests verify that the
artifact, metadata, registry version, and installed scikit-learn runtime agree.

`versions/` retains the active model and one structurally different Ridge rollback. The
retention policy is:

- keep the active checksummed version;
- keep one verified, diverse rollback version;
- remove superseded experimental versions only after the active artifact passes readiness,
  prediction, and checksum tests;
- update the registry before removing any registered files.

The Harris County model is a separate appraisal-value product and must not be described as
a verified sale-price model. Unregistered legacy pickle files were removed during the
August 2026 cleanup; see `../../reports/LEGACY_ARTIFACT_AUDIT.md` for the decision record.
