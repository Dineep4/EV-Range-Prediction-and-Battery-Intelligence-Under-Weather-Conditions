import pytest
from fastapi.testclient import TestClient
import app.main as api

client = TestClient(api.app)


def test_soh_predict_success_with_valid_inputs():
    response = client.post(
        "/api/predict/soh",
        json={
            "battery_id": "NASA-18650-Test",
            "cycle_index": 50,
            "voltage_mean": 3.53,
            "current_mean": -1.82,
            "temperature_mean": 32.5,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_soh_percent" in data
    assert 50.0 <= data["predicted_soh_percent"] <= 110.0
    assert data["model_name"] == "Gradient Boosting Regressor"
    assert "uncertainty_interval" in data
    assert data["uncertainty_interval"]["lower_bound_percent"] <= data["predicted_soh_percent"]
    assert data["uncertainty_interval"]["upper_bound_percent"] >= data["predicted_soh_percent"]
    assert "disclaimer" in data
    assert "not a substitute" in data["disclaimer"].lower()
    assert data["held_out_test_cell"] == "B0006"
    assert data["evaluation_metrics"]["r2_score"] > 0.85


def test_soh_predict_with_measured_capacity():
    response = client.post(
        "/api/predict/soh",
        json={
            "battery_id": "B0005",
            "cycle_index": 20,
            "voltage_mean": 3.55,
            "current_mean": -1.85,
            "temperature_mean": 32.4,
            "capacity_ah": 1.84,
            "rated_capacity_ah": 2.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["measured_soh_percent"] == 92.0
    assert abs(data["predicted_soh_percent"] - 92.0) < 15.0


def test_soh_predict_handles_positive_current_sign():
    response = client.post(
        "/api/predict/soh",
        json={
            "cycle_index": 80,
            "voltage_mean": 3.51,
            "current_mean": 1.82,  # positive sign normalized by service
            "temperature_mean": 33.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 50.0 <= data["predicted_soh_percent"] <= 110.0


def test_soh_predict_rejects_invalid_voltage_range():
    response = client.post(
        "/api/predict/soh",
        json={
            "cycle_index": 50,
            "voltage_mean": 15.0,  # exceeds maximum valid voltage (10.0 in schema)
            "current_mean": -1.8,
            "temperature_mean": 30.0,
        },
    )
    assert response.status_code == 422


def test_soh_metrics_endpoint_returns_held_out_performance():
    response = client.get("/api/soh/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["held_out_test_cell"] == "B0006"
    assert data["metrics"]["rmse_percent"] > 0
    assert data["metrics"]["r2_score"] > 0.85
    assert data["target_name"] == "soh_percent"
