# Model artifacts

The Django Seattle endpoint resolves its active artifact through `model_registry.json`. Version
3.0.0 is trained from privacy-safe King County Assessor extracts and bundles the preprocessing
pipeline plus a compact comparable-sale reference set. The registry is required: there is no
unregistered fallback artifact.

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
In October 2026 the superseded experimental `2.2.0` version and the `trained_models.pkl` /
`metadata.json` compatibility copies (byte-identical to `versions/3.0.0/`) were removed under the
policy above; they remain recoverable from Git history.

The deployed website does not load these files at runtime: `python -m
ml.training.export_browser_model` converts the active version to JSON that the React app runs in
the browser (see `../../frontend/README.md`).
