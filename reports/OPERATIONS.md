# Inference Operations

## Scope

The operational layer is deliberately lightweight and uses the existing Django database and
standard logging. It does not add a metrics cluster or make model inference depend on an
experiment tracker.

## Structured logging

The `api` logger emits one-line JSON records. Model loading and each completed inference include
an event name, request ID, endpoint, status, model version, latency, and error category where
applicable. Logs exclude request payloads, addresses, and external market-context responses.

Every prediction response contains the same UUID in its JSON `request_id` and `X-Request-ID`
header. A valid caller-supplied UUID is propagated; malformed values are replaced.

## Database telemetry

`PredictionLog` stores successful and failed events. Successful Seattle events retain only the
five model fields needed for feature-range monitoring. Failed inputs, Harris categorical fields,
addresses, and pincodes are not stored. The table records:

- timestamp, request ID, endpoint, status, and error category;
- model version and inference latency;
- prediction, interval, and input-confidence label when inference succeeds.

## Monitoring endpoint

`GET /monitoring/?days=7` accepts windows from 1 to 90 days and returns request volume, error
rate, mean and p95 latency, prediction distribution, model-version counts, error categories,
recent event metadata, and feature-range summaries. Configure `MONITORING_API_KEY` and pass it
through `X-Monitoring-Key` in any shared environment.

The feature signal is the fraction of requests outside each training feature's p05/p95 range.
It is useful for investigation, but is not a formal population-stability or hypothesis test.
No prediction-quality monitoring is possible until verified realized sale prices are collected.

## Artifact integrity and rollback

The active entry in `backend/mlmodels/model_registry.json` contains SHA-256 values for the model
and metadata files. These are checked before deserialization. Set `MODEL_VERSION` to another
registered version for rollback; restart web workers so their process caches load that version.

## Optional MLflow

MLflow is isolated in `requirements-mlflow.txt`. Passing `--track-mlflow` to the final training
command records parameters, metrics, dataset hash, artifact, reports, and final metadata. Normal
training does not enable it automatically, and Django never imports MLflow.
