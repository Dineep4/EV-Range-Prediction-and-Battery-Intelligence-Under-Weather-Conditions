"""API integration tests for the trip-distance prediction and model-performance endpoints."""

import pytest
from fastapi.testclient import TestClient

import app.main as api

client = TestClient(api.app)


def test_models_status_reports_trip_distance_available():
    res = client.get("/api/models/status")
    assert res.status_code == 200
    models = res.json()["models"]
    assert "trip_distance_research" in models
    info = models["trip_distance_research"]
    assert info["status"] == "available"
    assert "XGBoost" in info["models"]
    assert "LSTM" in info["models"]
    assert "GRU" in info["models"]
    assert "Hybrid CNN-LSTM" in info["models"]
    assert info["target"] == "total_trip_distance_km"
    assert info["is_research_estimate"] is True


def test_model_performance_includes_trip_distance_experiments():
    res = client.get("/api/model-performance")
    assert res.status_code == 200
    data = res.json()
    assert "trip_distance_research" in data
    research = data["trip_distance_research"]
    assert research["target"] == "total_trip_distance_km"
    assert research["observation_window_s"] == 30.0
    experiments = research["experiments"]
    assert len(experiments) == 5
    model_names = [e["model"] for e in experiments]
    assert "Training-Mean Baseline" in model_names
    assert "XGBoost" in model_names
    assert "LSTM" in model_names
    assert "GRU" in model_names
    assert "Hybrid CNN-LSTM" in model_names

    # Check metrics are valid numbers
    for exp in experiments:
        assert exp["rmse"] > 0
        assert exp["mae"] > 0
        assert exp["prediction_count"] == 12


def test_predict_trip_distance_xgboost_success():
    payload = {
        "model_name": "XGBoost",
        "window_distance_km": 0.28,
        "speed_mean": 35.0,
        "speed_max": 48.0,
        "speed_min": 0.0,
        "speed_std": 12.0,
        "power_mean": 10.5,
        "power_max": 28.0,
        "power_min": -5.0,
        "current_mean": 27.0,
        "voltage_mean": 382.0,
        "soc_start": 82.0,
        "soc_current": 81.8,
        "soc_delta": -0.2,
        "temperature_mean": 22.0,
    }
    res = client.post("/api/predict/trip-distance", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "VED Initial-Window EV Trip Distance Prediction"
    assert data["model"] == "XGBoost"
    assert data["target_units"] == "km"
    assert data["is_research_estimate"] is True
    assert isinstance(data["predicted_trip_distance_km"], float)
    assert data["predicted_trip_distance_km"] > 0


def test_predict_trip_distance_baseline():
    payload = {
        "model_name": "Training-Mean Baseline",
        "window_distance_km": 0.2,
        "speed_mean": 25.0,
        "speed_max": 30.0,
        "speed_min": 10.0,
        "speed_std": 5.0,
        "power_mean": 6.0,
        "power_max": 15.0,
        "power_min": 0.0,
        "current_mean": 15.0,
        "voltage_mean": 390.0,
        "soc_start": 75.0,
        "soc_current": 74.9,
        "soc_delta": -0.1,
        "temperature_mean": 18.0,
    }
    res = client.post("/api/predict/trip-distance", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "Training-Mean Baseline"
    assert round(data["predicted_trip_distance_km"], 2) == 2.60


def test_predict_trip_distance_neural_models():
    for model in ["LSTM", "GRU", "Hybrid CNN-LSTM"]:
        payload = {
            "model_name": model,
            "window_distance_km": 0.3,
            "speed_mean": 36.0,
            "speed_max": 50.0,
            "speed_min": 0.0,
            "speed_std": 10.0,
            "power_mean": 9.0,
            "power_max": 25.0,
            "power_min": -8.0,
            "current_mean": 24.0,
            "voltage_mean": 380.0,
            "soc_start": 80.0,
            "soc_current": 79.8,
            "soc_delta": -0.2,
            "temperature_mean": 20.0,
        }
        res = client.post("/api/predict/trip-distance", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["model"] == model
        assert isinstance(data["predicted_trip_distance_km"], float)
        assert data["predicted_trip_distance_km"] > 0


def test_predict_trip_distance_rejects_unknown_model():
    payload = {
        "model_name": "RandomForestRegressor",
        "window_distance_km": 0.2,
        "speed_mean": 25.0,
        "speed_max": 30.0,
        "speed_min": 10.0,
        "speed_std": 5.0,
        "power_mean": 6.0,
        "power_max": 15.0,
        "power_min": 0.0,
        "current_mean": 15.0,
        "voltage_mean": 390.0,
        "soc_start": 75.0,
        "soc_current": 74.9,
        "soc_delta": -0.1,
        "temperature_mean": 18.0,
    }
    res = client.post("/api/predict/trip-distance", json=payload)
    assert res.status_code == 422
    assert "Unsupported model" in res.json()["detail"]


def test_predict_trip_distance_rejects_invalid_values():
    payload = {
        "model_name": "XGBoost",
        "window_distance_km": -5.0,  # Invalid negative distance
        "speed_mean": 25.0,
        "speed_max": 30.0,
        "speed_min": 10.0,
        "speed_std": 5.0,
        "power_mean": 6.0,
        "power_max": 15.0,
        "power_min": 0.0,
        "current_mean": 15.0,
        "voltage_mean": 390.0,
        "soc_start": 75.0,
        "soc_current": 74.9,
        "soc_delta": -0.1,
        "temperature_mean": 18.0,
    }
    res = client.post("/api/predict/trip-distance", json=payload)
    assert res.status_code == 422
