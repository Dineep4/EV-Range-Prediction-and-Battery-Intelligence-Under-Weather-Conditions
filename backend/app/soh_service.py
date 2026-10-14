"""Isolated Battery State of Health (SOH) prediction service.

Provides data-leakage-free inference using the validated Gradient Boosting
model trained on NASA 2.0 Ah commercial 18650 cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException
import joblib
import numpy as np
from pydantic import BaseModel, Field


class SOHPredictionRequest(BaseModel):
    battery_id: str = Field(default="NASA-18650", max_length=50)
    cycle_index: float = Field(
        default=50.0,
        ge=1.0,
        le=500.0,
        description="Cycle count / operational cycle index (1-500)",
    )
    voltage_mean: float = Field(
        default=3.53,
        ge=2.0,
        le=4.5,
        description="Mean discharge operating voltage in Volts (2.0-4.5 V)",
    )
    current_mean: float = Field(
        default=-1.82,
        ge=-10.0,
        le=10.0,
        description="Mean discharge load current in Amperes (-10 to 10 A)",
    )
    temperature_mean: float = Field(
        default=32.5,
        ge=0.0,
        le=70.0,
        description="Mean cell operating temperature in °C (0-70 °C)",
    )
    capacity_ah: float | None = Field(
        default=None,
        gt=0.0,
        le=5.0,
        description="Optional measured full-charge capacity in Ah",
    )
    rated_capacity_ah: float | None = Field(
        default=2.0,
        gt=0.0,
        le=5.0,
        description="Reference / rated battery capacity in Ah (default: 2.0 Ah)",
    )


def load_soh_artifact(models_dir: Path) -> dict[str, Any]:
    artifact_path = models_dir / "battery_soh_model.joblib"
    if not artifact_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "SOH prediction is unavailable because no validated battery SOH "
                "model artifact is installed at models/battery_soh_model.joblib."
            ),
        )

    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load battery SOH model artifact: {exc}",
        ) from exc

    if not isinstance(artifact, dict) or artifact.get("validation_status") != "VALIDATED_SOH_MODEL":
        raise HTTPException(
            status_code=503,
            detail="The installed SOH artifact is unvalidated or missing required validation metadata.",
        )

    required_keys = ["model", "preprocessor", "feature_order", "metrics", "target_definition"]
    missing = [k for k in required_keys if k not in artifact]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"SOH artifact is missing required keys: {missing}",
        )

    return artifact


def run_soh_prediction(req: SOHPredictionRequest, models_dir: Path) -> dict[str, Any]:
    artifact = load_soh_artifact(models_dir)
    model = artifact["model"]
    scaler = artifact["preprocessor"]
    feature_order = artifact["feature_order"]

    # In NASA measurements, discharge current is negative. Normalize sign if user passed positive current.
    current_val = float(req.current_mean)
    if current_val > 0:
        current_val = -current_val

    feature_map = {
        "cycle_index": float(req.cycle_index),
        "voltage_mean": float(req.voltage_mean),
        "current_mean": current_val,
        "temperature_mean": float(req.temperature_mean),
    }

    raw_features = np.asarray([[feature_map[name] for name in feature_order]], dtype=np.float64)

    try:
        scaled_features = scaler.transform(raw_features)
        prediction_val = float(model.predict(scaled_features)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Battery SOH inference failed: {exc}",
        ) from exc

    if not np.isfinite(prediction_val):
        raise HTTPException(
            status_code=500,
            detail="Battery SOH model returned a non-finite prediction.",
        )

    predicted_soh = round(float(np.clip(prediction_val, 0.0, 120.0)), 2)

    # Calculate empirical prediction uncertainty interval from held-out RMSE
    rmse = float(artifact.get("uncertainty_rmse", 3.498))
    margin_90 = round(1.645 * rmse, 2)
    uncertainty_interval = {
        "lower_bound_percent": round(max(0.0, predicted_soh - margin_90), 2),
        "upper_bound_percent": round(min(120.0, predicted_soh + margin_90), 2),
        "margin_percent": margin_90,
        "confidence_level": "90% empirical prediction interval (±1.645 RMSE)",
    }

    # If measured capacity is provided, compute direct formula check
    measured_soh = None
    if req.capacity_ah is not None and req.capacity_ah > 0:
        rated_ref = req.rated_capacity_ah if (req.rated_capacity_ah and req.rated_capacity_ah > 0) else 2.0
        measured_soh = round(100.0 * float(req.capacity_ah) / float(rated_ref), 2)

    return {
        "status": "success",
        "predicted_soh_percent": predicted_soh,
        "uncertainty_interval": uncertainty_interval,
        "measured_soh_percent": measured_soh,
        "model_name": "Gradient Boosting Regressor",
        "model_architecture": artifact.get("model_architecture", "GradientBoostingRegressor"),
        "target": artifact.get("target", "soh_percent"),
        "target_definition": artifact.get("target_definition", "100 * measured discharge capacity / 2.0 Ah rated capacity"),
        "rated_capacity_ah": artifact.get("rated_capacity_ah", 2.0),
        "dataset_source": artifact.get("source_dataset", "NASA PCoE Li-ion Battery Aging Dataset"),
        "evaluation_metrics": artifact.get("metrics", {}),
        "held_out_test_cell": artifact.get("test_cell", "B0006"),
        "input_values": {
            "battery_id": req.battery_id,
            "cycle_index": req.cycle_index,
            "voltage_mean": req.voltage_mean,
            "current_mean": req.current_mean,
            "temperature_mean": req.temperature_mean,
            "capacity_ah": req.capacity_ah,
            "rated_capacity_ah": req.rated_capacity_ah,
        },
        "disclaimer": (
            "This is a data-driven estimate based on the selected laboratory battery "
            "model and available operating inputs. It is not a substitute for a "
            "certified battery diagnostic or battery-management-system measurement."
        ),
    }
