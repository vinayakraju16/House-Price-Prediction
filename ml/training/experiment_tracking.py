"""Optional MLflow adapter; normal training and inference do not depend on MLflow."""

from pathlib import Path


def log_mlflow_run(metadata, model_path, artifacts=(), tracking_uri=None,
                   experiment_name="house-price-seattle"):
    """Log a completed training run and mutate metadata with its MLflow run ID."""
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking was requested but MLflow is not installed. "
            "Install requirements-mlflow.txt first."
        ) from error

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"seattle-{metadata['model_version']}") as run:
        metadata["experiment_tracking"] = {
            "provider": "mlflow",
            "enabled": True,
            "experiment_name": experiment_name,
            "run_id": run.info.run_id,
        }
        params = {
            "model_type": metadata["model_type"],
            "target_transform": metadata["target_transform"]["forward"],
            "feature_count": metadata["feature_count"],
            **{f"model__{key}": value for key, value in metadata["hyperparameters"].items()},
        }
        mlflow.log_params(params)
        mlflow.log_metrics({
            "nested_cv_mae_mean": metadata["nested_cv"]["mae"]["mean"],
            "nested_cv_rmse_mean": metadata["nested_cv"]["rmse"]["mean"],
            "nested_cv_r2_mean": metadata["nested_cv"]["r2"]["mean"],
            "legacy_test_mae": metadata["metrics"]["mae"],
            "legacy_test_rmse": metadata["metrics"]["rmse"],
            "legacy_test_r2": metadata["metrics"]["r2"],
        })
        mlflow.set_tags({
            "model_version": metadata["model_version"],
            "status": metadata["status"],
            "dataset_train_sha256": metadata["dataset_version"]["train_sha256"],
        })
        mlflow.log_artifact(str(model_path), artifact_path="model")
        for artifact in artifacts:
            path = Path(artifact)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="reports")
        mlflow.log_dict(metadata, "metadata.json")
    return metadata["experiment_tracking"]
