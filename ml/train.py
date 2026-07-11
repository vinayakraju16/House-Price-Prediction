"""Train the deployable house-price ensemble from the selected feature data.

Run from the repository root:
    python ml/train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "final_14_selected_features.csv"
MODEL_PATH = ROOT / "backend" / "mlmodels" / "trained_models.pkl"
METADATA_PATH = ROOT / "backend" / "mlmodels" / "metadata.json"
TARGET = "SalePrice"


def main():
    data = pd.read_csv(DATA_PATH)
    features = data.drop(columns=[TARGET])
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    elastic_net = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=100, l1_ratio=0.9, max_iter=20_000, random_state=42)),
    ])
    gradient_boost = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    elastic_net.fit(x_train, y_train)
    gradient_boost.fit(x_train, y_train)

    elastic_predictions = elastic_net.predict(x_test)
    gradient_predictions = gradient_boost.predict(x_test)
    combined_predictions = (elastic_predictions + gradient_predictions) / 2
    metrics = {
        "elastic_net": _metrics(y_test, elastic_predictions),
        "gradient_boost": _metrics(y_test, gradient_predictions),
        "ensemble": _metrics(y_test, combined_predictions),
        "deployed_model": "gradient_boost",
        "features": list(features.columns),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "dataset_rows": len(data),
        "target_summary": {
            "minimum": round(float(target.min()), 2),
            "median": round(float(target.median()), 2),
            "maximum": round(float(target.max()), 2),
        },
        "feature_importance": _feature_importance(features.columns, gradient_boost.feature_importances_),
        "typical_feature_ranges": _typical_ranges(features),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"best_elastic_net": elastic_net, "best_gboost": gradient_boost}, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _metrics(actual, predicted):
    return {
        "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 2),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def _feature_importance(names, importances):
    ranked = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
    return [
        {"feature": name, "importance": round(float(importance), 4)}
        for name, importance in ranked
    ]


def _typical_ranges(features):
    return {
        column: {
            "minimum": round(float(features[column].astype(float).min()), 2),
            "p05": round(float(features[column].astype(float).quantile(0.05)), 2),
            "p95": round(float(features[column].astype(float).quantile(0.95)), 2),
            "maximum": round(float(features[column].astype(float).max()), 2),
        }
        for column in features.columns
    }


if __name__ == "__main__":
    main()
