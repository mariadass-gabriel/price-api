"""Train a baseline and an XGBoost model, track runs in MLflow, export the best one.

The exported artifact (models/model.joblib) holds the fitted pipeline plus the
split-conformal quantile. The API only loads that file; it never talks to MLflow.

Usage: python -m src.train
"""
import math
from pathlib import Path

import joblib
import mlflow
import numpy as np
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data import CATEGORICAL, FEATURES, NUMERIC, TARGET, TEXT, load_dataset

SEED = 42
ALPHA = 0.1  # 90 % prediction intervals
MODEL_PATH = Path("models/model.joblib")
TRACKING_URI = "sqlite:///mlflow.db"  # local file, no server
EXPERIMENT = "price-api"


def preprocessor(scale: bool) -> ColumnTransformer:
    # reviews is heavy-tailed: log-scale it; scaling only matters for the linear model.
    numeric = [FunctionTransformer(np.log1p, feature_names_out="one-to-one")]
    if scale:
        numeric.append(StandardScaler())
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", Pipeline([(f"n{i}", t) for i, t in enumerate(numeric)]), NUMERIC),
        # Product titles carry the capacity / power / model, the main price drivers.
        ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True), TEXT),
    ])


def log_target(regressor) -> TransformedTargetRegressor:
    """Prices are right-skewed: fit on log(1 + price), predict back in price units."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def candidates() -> dict:
    """Name -> (pipeline, params to log)."""
    ridge_params = {"alpha": 1.0}
    xgb_params = {
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "min_child_weight": 2,
        "random_state": SEED,
        "n_jobs": 1,
    }
    return {
        "ridge_baseline": (
            log_target(Pipeline([("prep", preprocessor(scale=True)), ("model", Ridge(**ridge_params))])),
            ridge_params,
        ),
        "xgboost": (
            log_target(Pipeline([("prep", preprocessor(scale=False)), ("model", XGBRegressor(**xgb_params))])),
            xgb_params,
        ),
    }


def conformal_quantile(abs_errors: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample correction ceil((n+1)(1-alpha))/n."""
    n = len(abs_errors)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(abs_errors, level, method="higher"))


def main() -> None:
    df = load_dataset()
    X, y = df[FEATURES], df[TARGET]

    # 60 % train / 15 % validation (model selection) / 15 % calibration / 10 % test
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=SEED)
    X_val, X_rest, y_val, y_rest = train_test_split(X_rest, y_rest, test_size=0.625, random_state=SEED)
    X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.4, random_state=SEED)
    print(f"rows: train={len(X_train)} val={len(X_val)} cal={len(X_cal)} test={len(X_test)}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    fitted, val_mae = {}, {}
    for name, (pipe, params) in candidates().items():
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)
            val_mae[name] = mean_absolute_error(y_val, pred)
            mlflow.log_param("model", name)
            mlflow.log_params(params)
            mlflow.log_metric("val_mae", val_mae[name])
            mlflow.log_metric("val_r2", r2_score(y_val, pred))
            fitted[name] = pipe
            print(f"{name}: val MAE={val_mae[name]:.2f}")

    best = min(val_mae, key=val_mae.get)
    pipe = fitted[best]

    # Split conformal: the calibration set was used neither for fitting nor for selection.
    q = conformal_quantile(np.abs(y_cal - pipe.predict(X_cal)), ALPHA)

    test_pred = pipe.predict(X_test)
    low, high = np.maximum(test_pred - q, 0), test_pred + q
    test_metrics = {
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_r2": r2_score(y_test, test_pred),
        "test_coverage": float(np.mean((y_test >= low) & (y_test <= high))),
        "conformal_quantile": q,
    }
    with mlflow.start_run(run_name=f"best_{best}"):
        mlflow.log_param("model", best)
        mlflow.log_param("alpha", ALPHA)
        mlflow.log_metrics(test_metrics)
    print(f"best={best} q90={q:.2f} " + " ".join(f"{k}={v:.3f}" for k, v in test_metrics.items()))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": pipe,
            "model_name": best,
            "conformal_quantile": q,
            "alpha": ALPHA,
            "features": FEATURES,
            "metrics": test_metrics,
            "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        },
        MODEL_PATH,
        compress=3,
    )
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
