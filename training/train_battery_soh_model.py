"""Train and evaluate an isolated, data-leakage-free Battery State of Health (SOH) regression model.

Dataset:
    NASA PCoE Li-ion Battery Aging Dataset (2.0 Ah nominal 18650 cylindrical cells).
    Source files processed in data/processed/battery_measurements.csv.
    Standard room-temperature and warm cells with validated 2.0 Ah rating:
    B0005, B0006, B0007, B0018, B0025, B0026, B0027, B0028 (748 total cycles).

Target Definition:
    SOH (%) = (Current measured discharge capacity / Documented rated capacity of 2.0 Ah) * 100

Partition Protocol:
    Strict cell-level holdout partition to prevent data leakage across cycles of the same cell.
    - Test partition: Cell B0006 held out strictly (168 cycles).
    - Training partition: Cells B0005, B0007, B0018, B0025, B0026, B0027, B0028 (580 cycles).
    Preprocessing (StandardScaler) is fit strictly on the training partition.

Features:
    - cycle_index: operational charge/discharge cycle number (integer >= 1)
    - voltage_mean: mean discharge operating voltage (V)
    - current_mean: mean discharge current (A)
    - temperature_mean: mean operating cell temperature (deg C)
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "battery_measurements.csv"
MODEL_OUTPUT_PATH = ROOT / "models" / "battery_soh_model.joblib"
METRICS_OUTPUT_PATH = ROOT / "results" / "battery_soh_metrics.json"

FEATURE_NAMES = [
    "cycle_index",
    "voltage_mean",
    "current_mean",
    "temperature_mean",
]

RATED_CAPACITY_AH = 2.0
CLEAN_CELLS = ["B0005", "B0006", "B0007", "B0018", "B0025", "B0026", "B0027", "B0028"]
HELD_OUT_TEST_CELL = "B0006"


def train_and_evaluate() -> dict:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Processed battery measurements not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    clean_df = df[df["battery_id"].isin(CLEAN_CELLS)].copy()
    
    # Calculate defensible SOH target relative to manufacturer 2.0 Ah rated capacity
    clean_df["soh_percent"] = 100.0 * clean_df["capacity_ah"] / RATED_CAPACITY_AH

    train_df = clean_df[clean_df["battery_id"] != HELD_OUT_TEST_CELL].copy()
    test_df = clean_df[clean_df["battery_id"] == HELD_OUT_TEST_CELL].copy()

    X_train = train_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_train = train_df["soh_percent"].to_numpy(dtype=np.float64)

    X_test = test_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_test = test_df["soh_percent"].to_numpy(dtype=np.float64)

    # Preprocessing strictly fit on training partition
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train primary Gradient Boosting model and baseline models
    candidate_models = {
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        ),
        "ridge_regression": Ridge(alpha=1.0),
    }

    eval_results = {}
    for name, model in candidate_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0)

        eval_results[name] = {
            "mae_percent": round(mae, 4),
            "rmse_percent": round(rmse, 4),
            "r2_score": round(r2, 4),
            "mape_percent": round(mape, 4),
        }

    # Selected production model is Gradient Boosting
    selected_name = "gradient_boosting"
    selected_model = candidate_models[selected_name]
    best_metrics = eval_results[selected_name]

    # Calculate feature summary stats for backend validation
    feature_constraints = {}
    for idx, feat in enumerate(FEATURE_NAMES):
        vals = train_df[feat].to_numpy(dtype=np.float64)
        feature_constraints[feat] = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
        }

    artifact = {
        "model": selected_model,
        "preprocessor": scaler,
        "feature_order": FEATURE_NAMES,
        "feature_constraints": feature_constraints,
        "target": "soh_percent",
        "target_definition": "100 * measured discharge capacity (Ah) / 2.0 Ah documented rated capacity",
        "rated_capacity_ah": RATED_CAPACITY_AH,
        "training_cells": sorted(list(set(train_df["battery_id"]))),
        "training_sample_count": len(train_df),
        "test_cell": HELD_OUT_TEST_CELL,
        "test_sample_count": len(test_df),
        "metrics": best_metrics,
        "all_model_evaluations": eval_results,
        "uncertainty_rmse": best_metrics["rmse_percent"],
        "training_status": "TRAINED_AND_EVALUATED",
        "validation_status": "VALIDATED_SOH_MODEL",
        "model_architecture": "GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.05)",
        "source_dataset": "NASA PCoE Li-ion Battery Aging Dataset (18650 cells)",
    }

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print(f"Saved SOH model artifact to: {MODEL_OUTPUT_PATH}")

    metrics_payload = {
        "model_name": "Gradient Boosting Regressor",
        "task": "Battery State of Health (SOH) Estimation",
        "dataset": "NASA PCoE Li-ion Battery Aging Dataset",
        "cell_type": "2.0 Ah commercial 18650 Li-ion cells",
        "target_name": "soh_percent",
        "target_definition": "100 * measured discharge capacity (Ah) / 2.0 Ah rated capacity",
        "rated_capacity_ah": RATED_CAPACITY_AH,
        "held_out_test_cell": HELD_OUT_TEST_CELL,
        "test_samples": len(test_df),
        "train_samples": len(train_df),
        "metrics": best_metrics,
        "candidate_models": eval_results,
        "feature_names": FEATURE_NAMES,
        "uncertainty_margin_90_percent": round(1.645 * best_metrics["rmse_percent"], 2),
        "disclaimer": "Laboratory cell degradation model based on NASA 18650 cells under CC-CV cycle aging. Not a certified BMS diagnostic.",
    }

    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved SOH evaluation metrics to: {METRICS_OUTPUT_PATH}")

    return metrics_payload


if __name__ == "__main__":
    result = train_and_evaluate()
    print("Training Complete!")
    print(json.dumps(result, indent=2))
