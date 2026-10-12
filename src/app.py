"""Price prediction API. Loads models/model.joblib once at startup (no MLflow here)."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from src.schemas import HealthResponse, PredictRequest, PredictResponse

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))


def load_artifact(path: Path = MODEL_PATH) -> dict:
    """Load the dict written by src/train.py: fitted pipeline + conformal quantile."""
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifact = load_artifact()
    yield


app = FastAPI(title="Price API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=app.state.artifact["model_name"])


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    artifact = app.state.artifact
    features = pd.DataFrame([request.model_dump()])[artifact["features"]]
    # Prices cannot be negative, so the prediction itself is floored at 0.
    price = max(float(artifact["model"].predict(features)[0]), 0.0)
    q = artifact["conformal_quantile"]
    return PredictResponse(
        price=round(price, 2),
        low=round(max(price - q, 0.0), 2),
        high=round(price + q, 2),
        coverage=1 - artifact["alpha"],
    )
