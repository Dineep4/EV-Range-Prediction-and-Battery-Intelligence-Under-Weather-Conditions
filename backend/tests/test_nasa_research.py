import json
import math
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as api

client = TestClient(api.app)


def test_registered_nasa_xgboost_predicts_with_exact_proxy_metadata():
    response = client.post(
        "/api/predict/nasa-capacity-retention",
        json={"cycle_index_in_file": 180},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["config_id"] == "nasa_fy08q4:nasa_first_positive:xgboost"
    assert result["input_features"] == ["cycle_index_in_file"]
    assert result["input_values"] == {"cycle_index_in_file": 180}
    assert result["target"] == "first_positive_measurement_capacity_retention_percent"
    assert result["unit"] == "% capacity retention"
    assert result["provisional_proxy"] is True
    assert result["personalized_to_individual_battery"] is False
    assert math.isfinite(result["value"])


def test_nasa_request_rejects_missing_extra_noninteger_and_out_of_training_range_inputs():
    requests = [
        {},
        {"cycle_index_in_file": 2, "capacity_ah": 2.0},
        {"cycle_index_in_file": 1},
        {"cycle_index_in_file": 615},
        {"cycle_index_in_file": 12.5},
        {"cycle_index_in_file": "180"},
        {"cycle_index": 180},
    ]
    for payload in requests:
        response = client.post("/api/predict/nasa-capacity-retention", json=payload)
        assert response.status_code == 422, (payload, response.text)


def test_registry_and_artifact_are_explicit_and_no_legacy_fallback_is_used():
    config = json.loads(api.NASA_RESEARCH_REGISTRY.read_text(encoding="utf-8"))
    assert config["config_id"] == "nasa_fy08q4:nasa_first_positive:xgboost"
    assert config["feature_order"] == ["cycle_index_in_file"]
    assert config["model"]["artifact_path"] == "models/nasa_first_positive_retention_xgboost.json"
    loaded, model = api._load_nasa_retention_model()
    assert loaded["feature_order"] == ["cycle_index_in_file"]
    assert model.get_booster().num_features() == 1
    assert (api.ROOT / config["model"]["artifact_path"]).is_file()


def test_missing_or_malformed_nasa_artifact_fails_clearly_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    # A legacy root-level artifact is deliberately insufficient for this registered contract.
    (model_dir / "xgboost_soh.joblib").write_text("not a NASA model", encoding="utf-8")
    missing = client.post("/api/predict/nasa-capacity-retention", json={"cycle_index_in_file": 180})
    assert missing.status_code == 503
    assert missing.json()["detail"]["code"] == "nasa_model_artifact_unavailable"
    (model_dir / "nasa_first_positive_retention_xgboost.json").write_text("malformed", encoding="utf-8")
    malformed = client.post("/api/predict/nasa-capacity-retention", json={"cycle_index_in_file": 180})
    assert malformed.status_code == 503
    assert malformed.json()["detail"]["code"] == "nasa_model_load_failed"


def test_model_status_distinguishes_research_proxy_from_physical_soh():
    status = client.get("/api/models/status")
    assert status.status_code == 200
    models = status.json()["models"]
    assert models["nasa_capacity_retention_proxy"]["status"] == "available"
    assert models["battery_soh"]["status"] == "unavailable"
    assert "provisional" in models["battery_soh"]["reason"]


def test_served_nasa_metrics_match_saved_grouped_evaluation_and_baseline():
    response = client.get("/api/model-performance")
    assert response.status_code == 200
    nasa = response.json()["nasa_capacity_retention"]
    evaluation = nasa["evaluation"]
    assert evaluation["test_batteries"] == ["B0006"]
    assert evaluation["test_rows"] == 149
    model = evaluation["model_results"]["xgboost"]
    baseline = evaluation["model_results"]["linear_cycle_trend_baseline_paired"]
    assert model["rmse_percentage_points"] == 8.480971318193465
    assert baseline["rmse_percentage_points"] == 7.073485221517558
    assert model["rmse_percentage_points"] > baseline["rmse_percentage_points"]
    assert model["n"] == baseline["n"] == 149


def test_nasa_four_model_comparison_reported():
    response = client.get("/api/model-performance")
    assert response.status_code == 200
    nasa = response.json()["nasa_capacity_retention"]
    results = nasa["evaluation"]["model_results"]
    for expected in ["xgboost", "lstm", "gru", "cnn_lstm", "linear_cycle_trend_baseline_paired"]:
        assert expected in results, f"Missing {expected}"
        m = results[expected]
        assert m["n"] == 149
        assert m["rmse_percentage_points"] > 0
        assert m["mape_percent"] > 0
        assert m["mae_percentage_points"] > 0
        assert "r2" in m

