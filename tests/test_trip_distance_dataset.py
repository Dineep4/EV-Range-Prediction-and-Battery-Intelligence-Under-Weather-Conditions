"""Unit tests for the trip-distance data-preparation pipeline and eligibility rules.

Covers:
- Timestamp ordering and unit conversion
- Speed-distance trapezoidal integration
- Missing intervals and quality thresholds
- Resampling and 30-second observation window extraction
- Exclusion of future telemetry and target leakage prevention
- Partition separation across vehicle identities
"""

import math
import numpy as np
import pandas as pd
import pytest

from scripts.prepare_trip_distance_dataset import (
    WINDOW_DURATION_S,
    WINDOW_STEPS,
    create_split_manifest,
    finite_number,
    load_eligible_trips,
    process_trip_window,
)


def make_telemetry_point(t_offset_ms: float, speed_kmh: float = 30.0, current: float = -15.0, soc: float = 85.0):
    return {
        "source_file": "VED_sample.csv",
        "vehicle_id": "10",
        "trip_id": "999",
        "timestamp_ms": 1000000.0 + t_offset_ms,
        "speed_kmh": speed_kmh,
        "battery_current_a": current,
        "battery_voltage_v": 380.0,
        "soc_percent": soc,
        "outside_air_temperature_c": 18.5,
    }


def test_finite_number_parsing():
    assert finite_number(12.5) == 12.5
    assert finite_number("42") == 42.0
    assert finite_number("   ") is None
    assert finite_number(None) is None
    assert finite_number("invalid") is None
    assert finite_number(float("nan")) is None
    assert finite_number(float("inf")) is None


def test_load_eligible_trips_filters_coverage_and_duration(tmp_path):
    csv_file = tmp_path / "trips.csv"
    data = pd.DataFrame([
        # Eligible: coverage >= 0.95, duration >= 30, distance > 0
        {
            "vehicle_id": "10",
            "trip_id": "1",
            "source_file": "f1.csv",
            "trip_duration_s": 50.0,
            "speed_integrated_distance_km": 1.5,
            "speed_distance_coverage_fraction": 0.96,
        },
        # Ineligible: coverage < 0.95
        {
            "vehicle_id": "10",
            "trip_id": "2",
            "source_file": "f1.csv",
            "trip_duration_s": 50.0,
            "speed_integrated_distance_km": 1.5,
            "speed_distance_coverage_fraction": 0.92,
        },
        # Ineligible: duration < 30s
        {
            "vehicle_id": "455",
            "trip_id": "3",
            "source_file": "f2.csv",
            "trip_duration_s": 25.0,
            "speed_integrated_distance_km": 0.5,
            "speed_distance_coverage_fraction": 0.98,
        },
        # Ineligible: distance <= 0
        {
            "vehicle_id": "455",
            "trip_id": "4",
            "source_file": "f2.csv",
            "trip_duration_s": 60.0,
            "speed_integrated_distance_km": 0.0,
            "speed_distance_coverage_fraction": 0.99,
        },
    ])
    data.to_csv(csv_file, index=False)

    eligible = load_eligible_trips(csv_file, min_coverage=0.95)
    assert len(eligible) == 1
    assert eligible.iloc[0]["trip_id"] == "1"


def test_process_trip_window_generates_correct_dimensions_and_no_leakage():
    # 40 seconds of points at 1-second intervals (0ms to 40000ms)
    points = [make_telemetry_point(i * 1000.0, speed_kmh=36.0) for i in range(41)]

    seq_matrix, tabular, qmeta = process_trip_window(points, window_duration_s=30.0, num_steps=30)

    # Dimensionality
    assert seq_matrix.shape == (30, 7)  # 30 steps x 7 features
    assert np.isfinite(seq_matrix).all()

    # Speed was constant 36 km/h: in 29 seconds of travel on grid,
    # distance = 36 km/h * (29 / 3600 h) = 0.29 km
    assert tabular["speed_mean"] == pytest.approx(36.0, rel=1e-3)
    assert tabular["window_distance_km"] == pytest.approx(0.29, rel=1e-2)
    assert tabular["window_duration_s"] == 30.0
    assert qmeta["window_coverage_valid"] is True

    # Check that points beyond 30s were NOT used in window_distance
    # If 40s had leaked, distance would be 0.40 km
    assert tabular["window_distance_km"] < 0.35


def test_process_trip_window_rejects_trip_shorter_than_window():
    # Trip is only 20 seconds long
    points = [make_telemetry_point(i * 1000.0) for i in range(21)]
    with pytest.raises(ValueError, match="less than window"):
        process_trip_window(points, window_duration_s=30.0, num_steps=30)


def test_create_split_manifest_vehicle_isolation():
    # Create mock 51 eligible trips: 12 for veh 10, 38 for veh 455, 1 for veh 541
    records = []
    for i in range(12):
        records.append({"vehicle_id": "10", "trip_id": str(1000 + i)})
    for i in range(38):
        records.append({"vehicle_id": "455", "trip_id": str(2000 + i)})
    records.append({"vehicle_id": "541", "trip_id": "3001"})

    mock_df = pd.DataFrame(records)
    manifest = create_split_manifest(mock_df, seed=42)

    assert manifest["test_count"] == 12
    assert manifest["validation_count"] == 8
    assert manifest["train_count"] == 31
    assert manifest["test_vehicle_ids"] == ["10"]

    test_keys = set(manifest["test_trip_keys"])
    val_keys = set(manifest["validation_trip_keys"])
    train_keys = set(manifest["train_trip_keys"])

    # Disjoint check
    assert len(test_keys & val_keys) == 0
    assert len(test_keys & train_keys) == 0
    assert len(val_keys & train_keys) == 0
    assert len(test_keys | val_keys | train_keys) == 51

    # Vehicle 10 must NEVER appear in train or validation
    assert all(k.startswith("10_") for k in test_keys)
    assert not any(k.startswith("10_") for k in val_keys)
    assert not any(k.startswith("10_") for k in train_keys)
