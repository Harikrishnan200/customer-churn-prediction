"""
Tests for the FastAPI app (app/main.py).

We swap `app.main.predictor` for a small fake object rather than
requiring a real MLflow server to be running during tests. This keeps
the test suite fast and able to run in CI, where no MLflow server
exists. The fake behaves exactly like ChurnPredictor from the outside
(is_ready / predict), which is all app/main.py relies on.
"""

from fastapi.testclient import TestClient

import app.main as main_module

client = TestClient(main_module.app)


class FakePredictor:
    model_name = "churn-model"
    model_version = "1"
    model_alias = "production"

    def is_ready(self) -> bool:
        return True

    def predict(self, request: dict):
        return True, 0.9

VALID_PAYLOAD = {
    "tenure": 12,
    "monthly_charges": 75.5,
    "total_charges": 850.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "senior_citizen": 0,
    "partner": 1,
    "dependents": 0,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_rejects_invalid_input():
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["tenure"] = "not-a-number"

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete_payload = dict(VALID_PAYLOAD)
    del incomplete_payload["contract"]

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main_module, "predictor", None)

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_predict_success_with_loaded_model(monkeypatch):
    monkeypatch.setattr(main_module, "predictor", FakePredictor())

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["churn"] is True
    assert body["probability"] == 0.9
    assert body["model_version"] == "1"


def test_model_info_success_with_loaded_model(monkeypatch):
    monkeypatch.setattr(main_module, "predictor", FakePredictor())

    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "churn-model",
        "model_version": "1",
        "model_alias": "production",
    }


def test_model_info_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main_module, "predictor", None)

    response = client.get("/model-info")
    assert response.status_code == 503


def test_metrics_endpoint_exposes_prometheus_format():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.text
