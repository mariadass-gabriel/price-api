import pytest
from fastapi.testclient import TestClient

from src.app import MODEL_PATH, app, load_artifact

VALID = {
    "company_name": "pccomponentes",
    "brand_name": "Gigabyte",
    "category": "Discos Duros",
    "name": "Gigabyte SSD M.2 512GB 2280 PCIe x2 NVMe",
    "score": 9.5,
    "reviews": 40,
}


@pytest.fixture(scope="module")
def client():
    # The context manager runs the lifespan, which loads the model.
    with TestClient(app) as c:
        yield c


def test_model_loads():
    artifact = load_artifact(MODEL_PATH)
    assert {"model", "conformal_quantile", "alpha", "features"} <= artifact.keys()
    assert artifact["conformal_quantile"] > 0
    assert 0 < artifact["alpha"] < 1


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"price", "low", "high", "coverage"}
    assert body["price"] >= 0


def test_predict_interval_contains_prediction(client):
    body = client.post("/predict", json=VALID).json()
    assert body["low"] <= body["price"] <= body["high"]


def test_predict_unseen_category_still_works(client):
    r = client.post("/predict", json={**VALID, "brand_name": "NeverSeenBrand"})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in VALID.items() if k != "score"},  # missing field
        {**VALID, "score": 11},  # out of range
        {**VALID, "reviews": -1},  # negative
        {**VALID, "score": "high"},  # wrong type
        {**VALID, "category": ""},  # empty string
        {**VALID, "name": ""},  # empty product name
        {**VALID, "unexpected": 1},  # extra field
    ],
)
def test_predict_invalid_returns_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_cors_allows_portfolio_origin(client):
    origin = "https://mariadass-gabriel.github.io"
    r = client.options(
        "/predict",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == origin
    assert "access-control-allow-origin" not in client.options(
        "/predict",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    ).headers
