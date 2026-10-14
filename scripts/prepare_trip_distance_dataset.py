"""Prepare VED pure-EV trip distance dataset with a fixed initial observation window.

This script extracts raw telemetry for the 51 eligible pure-EV trips in VED
(vehicles 10, 455, 541 with speed-distance coverage >= 95%), constructs a fixed
initial 30-second observation window sampled at 1 Hz, and derives tabular summary
features and sequence tensors for four-model evaluation.

Target: Total trip distance in kilometres (speed_integrated_distance_km).
Excluded: Any telemetry after the 30-second window, completed-trip duration,
final SOC, final energy, and the target itself.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "VED"
STATIC_FILE = RAW_DIR / "VED_Static_Data_PHEV_EV.xlsx"
ARCHIVES = [
    RAW_DIR / "VED_DynamicData_Part1.7z",
    RAW_DIR / "VED_DynamicData_Part2.7z",
]
TRIPS_SUMMARY_CSV = ROOT / "data" / "processed" / "ved" / "ved_ev_trip_measurements.csv"

OUTPUT_RAW_DIR = ROOT / "data" / "raw" / "trip_distance"
OUTPUT_PROCESSED_DIR = ROOT / "data" / "processed" / "ved"
OUTPUT_RAW_TELEMETRY = OUTPUT_RAW_DIR / "ved_eligible_raw_telemetry.csv"
OUTPUT_DATASET_CSV = OUTPUT_PROCESSED_DIR / "ved_trip_distance_dataset.csv"
OUTPUT_SEQUENCES_NPZ = OUTPUT_PROCESSED_DIR / "ved_trip_distance_sequences.npz"
OUTPUT_SPLIT_MANIFEST = OUTPUT_PROCESSED_DIR / "ved_trip_distance_split_manifest.json"
OUTPUT_QUALITY_REPORT = OUTPUT_PROCESSED_DIR / "ved_trip_distance_quality_report.json"
OUTPUT_TARGET_DEFINITION = OUTPUT_PROCESSED_DIR / "ved_trip_distance_target_definition.json"

WINDOW_DURATION_S = 30.0
WINDOW_STEPS = 30  # 0 to 29 seconds (1 Hz)
MAX_GAP_S = 1.0

SEQUENCE_FEATURE_NAMES = [
    "speed_kmh",
    "battery_current_a",
    "battery_voltage_v",
    "power_kw",
    "soc_percent",
    "outside_air_temperature_c",
    "cumulative_window_distance_km",
]

TABULAR_FEATURE_NAMES = [
    "window_duration_s",
    "window_distance_km",
    "speed_mean",
    "speed_max",
    "speed_min",
    "speed_std",
    "power_mean",
    "power_max",
    "power_min",
    "current_mean",
    "voltage_mean",
    "soc_start",
    "soc_current",
    "soc_delta",
    "temperature_mean",
]


def finite_number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return num if math.isfinite(num) else None


def load_eligible_trips(csv_path: Path, min_coverage: float = 0.95) -> pd.DataFrame:
    """Load trip summaries and select pure-EV trips meeting the coverage rule."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing trip summaries file: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {
        "vehicle_id",
        "trip_id",
        "source_file",
        "trip_duration_s",
        "speed_integrated_distance_km",
        "speed_distance_coverage_fraction",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Trip summaries missing columns: {missing}")

    eligible = df[
        (df["speed_distance_coverage_fraction"] >= min_coverage)
        & (df["trip_duration_s"] >= WINDOW_DURATION_S)
        & (df["speed_integrated_distance_km"] > 0)
    ].copy()
    eligible["vehicle_id"] = eligible["vehicle_id"].astype(str)
    eligible["trip_id"] = eligible["trip_id"].astype(str)
    return eligible


def stream_raw_telemetry(
    tar_exe: str,
    archive: Path,
    member: str,
    target_trips_by_veh: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Stream and filter raw telemetry points for specific trips from a 7z member."""
    proc = subprocess.Popen(
        [tar_exe, "-xOf", str(archive), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
    )
    records = []
    try:
        reader = csv.DictReader(proc.stdout)
        for row in reader:
            veh_raw = row.get("VehId", "").strip()
            # Normalize e.g. "10" or "10.0"
            veh_num = finite_number(veh_raw)
            veh_id = str(int(veh_num)) if veh_num is not None and veh_num.is_integer() else str(veh_raw)
            if veh_id not in target_trips_by_veh:
                continue
            trip_id = (row.get("Trip") or "").strip()
            if trip_id not in target_trips_by_veh[veh_id]:
                continue
            records.append({
                "source_file": member,
                "vehicle_id": veh_id,
                "trip_id": trip_id,
                "timestamp_ms": finite_number(row.get("Timestamp(ms)")),
                "speed_kmh": finite_number(row.get("Vehicle Speed[km/h]")),
                "battery_current_a": finite_number(row.get("HV Battery Current[A]")),
                "battery_voltage_v": finite_number(row.get("HV Battery Voltage[V]")),
                "soc_percent": finite_number(row.get("HV Battery SOC[%]")),
                "outside_air_temperature_c": finite_number(row.get("OAT[DegC]")),
            })
        stderr = proc.stderr.read()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Error reading {member} from {archive}: {stderr.strip()}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    return records


def process_trip_window(
    points: list[dict[str, Any]],
    window_duration_s: float = WINDOW_DURATION_S,
    num_steps: int = WINDOW_STEPS,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    """Process points for one trip to construct the 30-second observation window.

    Returns:
        (sequence_matrix, tabular_features, quality_meta)
    """
    valid_points = [p for p in points if p["timestamp_ms"] is not None]
    valid_points.sort(key=lambda p: p["timestamp_ms"])
    if not valid_points:
        raise ValueError("No points with valid timestamp.")

    t0 = valid_points[0]["timestamp_ms"]
    t_end = t0 + window_duration_s * 1000.0

    # Points within or just bounding the window
    window_points = [p for p in valid_points if p["timestamp_ms"] <= t_end]
    last_point = valid_points[-1]["timestamp_ms"]
    if last_point < t_end:
        raise ValueError(
            f"Trip duration { (last_point - t0)/1000:.1f}s is less than window {window_duration_s}s"
        )

    # Check for order violations and max gap in the window
    timestamps = [p["timestamp_ms"] for p in window_points]
    max_gap_s = 0.0
    for left, right in zip(timestamps, timestamps[1:]):
        gap = (right - left) / 1000.0
        if gap > max_gap_s:
            max_gap_s = gap

    # Resample to uniform 1 Hz grid (t = 0, 1, ..., num_steps - 1)
    # Using linear interpolation between adjacent points
    grid_times_ms = [t0 + i * 1000.0 for i in range(num_steps)]

    # Interpolate fields
    interpolated_rows = []
    cum_dist_km = 0.0

    for step_idx, target_t in enumerate(grid_times_ms):
        # Find bracketing points
        prev_p = None
        next_p = None
        for p in valid_points:
            if p["timestamp_ms"] <= target_t:
                prev_p = p
            if p["timestamp_ms"] >= target_t:
                next_p = p
                break

        if prev_p is None:
            prev_p = valid_points[0]
        if next_p is None:
            next_p = valid_points[-1]

        t_prev = prev_p["timestamp_ms"]
        t_next = next_p["timestamp_ms"]

        if t_next == t_prev:
            alpha = 0.0
        else:
            alpha = (target_t - t_prev) / (t_next - t_prev)

        def interp(key: str, default: float = 0.0) -> float:
            v_prev = prev_p.get(key)
            v_next = next_p.get(key)
            if v_prev is not None and v_next is not None:
                return float(v_prev + alpha * (v_next - v_prev))
            if v_prev is not None:
                return float(v_prev)
            if v_next is not None:
                return float(v_next)
            return default

        speed = max(0.0, interp("speed_kmh", 0.0))
        current = interp("battery_current_a", 0.0)
        voltage = max(0.0, interp("battery_voltage_v", 360.0))
        power = (current * voltage) / 1000.0
        soc = interp("soc_percent", 50.0)
        temp = interp("outside_air_temperature_c", 20.0)

        # Trapezoidal speed distance integration over 1-second grid
        if step_idx > 0:
            prev_speed = interpolated_rows[-1][0]
            step_km = (prev_speed + speed) * 0.5 * (1.0 / 3600.0)
            cum_dist_km += step_km

        interpolated_rows.append([
            speed,
            current,
            voltage,
            power,
            soc,
            temp,
            cum_dist_km,
        ])

    seq_matrix = np.asarray(interpolated_rows, dtype=np.float32)

    # Compute tabular summaries from the window
    speeds = seq_matrix[:, 0]
    powers = seq_matrix[:, 3]
    currents = seq_matrix[:, 1]
    voltages = seq_matrix[:, 2]
    socs = seq_matrix[:, 4]
    temps = seq_matrix[:, 5]

    tabular = {
        "window_duration_s": window_duration_s,
        "window_distance_km": float(cum_dist_km),
        "speed_mean": float(np.mean(speeds)),
        "speed_max": float(np.max(speeds)),
        "speed_min": float(np.min(speeds)),
        "speed_std": float(np.std(speeds)),
        "power_mean": float(np.mean(powers)),
        "power_max": float(np.max(powers)),
        "power_min": float(np.min(powers)),
        "current_mean": float(np.mean(currents)),
        "voltage_mean": float(np.mean(voltages)),
        "soc_start": float(socs[0]),
        "soc_current": float(socs[-1]),
        "soc_delta": float(socs[-1] - socs[0]),
        "temperature_mean": float(np.mean(temps)),
    }

    quality_meta = {
        "raw_points_in_window": len(window_points),
        "max_window_gap_s": float(max_gap_s),
        "window_coverage_valid": bool(max_gap_s <= MAX_GAP_S),
    }

    return seq_matrix, tabular, quality_meta


def create_split_manifest(
    trips_df: pd.DataFrame,
    seed: int = 42,
    val_from_455_count: int = 7,
) -> dict[str, Any]:
    """Create split manifest grouping by vehicle identity.

    Partition strategy:
    - Test: Vehicle 10 (12 trips) -> completely unseen vehicle identity.
    - Validation: Vehicle 541 (1 trip) + 7 deterministic trips from Vehicle 455 (8 trips total).
    - Train: Remaining 31 trips from Vehicle 455.
    """
    veh10_trips = sorted(trips_df[trips_df["vehicle_id"] == "10"]["trip_id"].tolist())
    veh541_trips = sorted(trips_df[trips_df["vehicle_id"] == "541"]["trip_id"].tolist())
    veh455_trips = sorted(trips_df[trips_df["vehicle_id"] == "455"]["trip_id"].tolist())

    rng = np.random.default_rng(seed)
    shuffled_455 = list(veh455_trips)
    rng.shuffle(shuffled_455)

    val_455 = sorted(shuffled_455[:val_from_455_count])
    train_455 = sorted(shuffled_455[val_from_455_count:])

    test_trips = [f"10_{tid}" for tid in veh10_trips]
    val_trips = [f"541_{tid}" for tid in veh541_trips] + [f"455_{tid}" for tid in val_455]
    train_trips = [f"455_{tid}" for tid in train_455]

    # Verify partitions
    all_keys = set(test_trips) | set(val_trips) | set(train_trips)
    assert len(test_trips) == 12, f"Expected 12 test trips, got {len(test_trips)}"
    assert len(val_trips) == 8, f"Expected 8 val trips, got {len(val_trips)}"
    assert len(train_trips) == 31, f"Expected 31 train trips, got {len(train_trips)}"
    assert len(all_keys) == 51, f"Expected 51 unique trips, got {len(all_keys)}"
    assert not (set(test_trips) & set(val_trips)), "Test and val trip overlap!"
    assert not (set(test_trips) & set(train_trips)), "Test and train trip overlap!"
    assert not (set(val_trips) & set(train_trips)), "Val and train trip overlap!"

    return {
        "split_strategy": (
            "Vehicle-grouped holdout: Vehicle 10 held out as unseen test vehicle (12 trips); "
            "Vehicle 541 (1 trip) and 7 trips of Vehicle 455 assigned to validation (8 trips); "
            "Remaining 31 trips of Vehicle 455 assigned to training."
        ),
        "seed": seed,
        "total_eligible_trips": 51,
        "train_count": len(train_trips),
        "validation_count": len(val_trips),
        "test_count": len(test_trips),
        "train_vehicle_ids": ["455"],
        "validation_vehicle_ids": ["455", "541"],
        "test_vehicle_ids": ["10"],
        "train_trip_keys": train_trips,
        "validation_trip_keys": val_trips,
        "test_trip_keys": test_trips,
        "limitation_note": (
            "All three vehicles are 2013 Nissan Leafs. Generalization across different EV makes "
            "or battery architectures cannot be established from this dataset."
        ),
    }


def main():
    print("Step 1: Loading eligible pure-EV trips from VED summaries...")
    eligible = load_eligible_trips(TRIPS_SUMMARY_CSV, min_coverage=0.95)
    print(f"Loaded {len(eligible)} eligible trips across vehicles: {eligible['vehicle_id'].value_counts().to_dict()}")

    tar_exe = shutil.which("tar")
    if not tar_exe:
        sys.exit("Error: System tar utility is required to stream VED archives.")

    OUTPUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Group eligible trips by source_file
    files_to_trips: dict[str, dict[str, set[str]]] = {}
    for _, row in eligible.iterrows():
        src = row["source_file"]
        veh = row["vehicle_id"]
        trip = row["trip_id"]
        files_to_trips.setdefault(src, {}).setdefault(veh, set()).add(trip)

    print(f"Step 2: Streaming raw telemetry across {len(files_to_trips)} source files...")
    all_raw_points: list[dict[str, Any]] = []

    for archive in ARCHIVES:
        if not archive.exists():
            sys.exit(f"Archive missing: {archive}")
        # List members
        res = subprocess.run([tar_exe, "-tf", str(archive)], capture_output=True, text=True, check=True)
        members = [m for m in res.stdout.splitlines() if m.lower().endswith(".csv")]
        for member in members:
            fname = member.rsplit("/", 1)[-1]
            if fname in files_to_trips:
                target_map = files_to_trips[fname]
                points = stream_raw_telemetry(tar_exe, archive, member, target_map)
                all_raw_points.extend(points)
                print(f"  Extracted {len(points)} raw points from {fname}")

    print(f"Total raw points extracted: {len(all_raw_points)}")

    # Save extracted raw telemetry
    print(f"Saving extracted raw telemetry to {OUTPUT_RAW_TELEMETRY}...")
    raw_df = pd.DataFrame(all_raw_points)
    raw_df.to_csv(OUTPUT_RAW_TELEMETRY, index=False)

    print("Step 3: Processing 30-second observation windows...")
    grouped = {}
    for p in all_raw_points:
        key = (p["vehicle_id"], p["trip_id"])
        grouped.setdefault(key, []).append(p)

    dataset_rows = []
    sequence_tensors = []
    trip_keys = []
    quality_records = []

    for _, row in eligible.iterrows():
        veh = row["vehicle_id"]
        trip = row["trip_id"]
        key = (veh, trip)
        trip_key = f"{veh}_{trip}"
        points = grouped.get(key, [])
        if not points:
            raise RuntimeError(f"No points found for eligible trip {trip_key}")

        seq_matrix, tabular, qmeta = process_trip_window(points, WINDOW_DURATION_S, WINDOW_STEPS)

        record = {
            "trip_key": trip_key,
            "vehicle_id": veh,
            "trip_id": trip,
            "total_trip_distance_km": float(row["speed_integrated_distance_km"]),
            "trip_duration_s": float(row["trip_duration_s"]),
            "speed_distance_coverage_fraction": float(row["speed_distance_coverage_fraction"]),
            **tabular,
        }
        dataset_rows.append(record)
        sequence_tensors.append(seq_matrix)
        trip_keys.append(trip_key)
        quality_records.append({
            "trip_key": trip_key,
            "vehicle_id": veh,
            "trip_id": trip,
            "total_trip_distance_km": float(row["speed_integrated_distance_km"]),
            **qmeta,
        })

    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df.to_csv(OUTPUT_DATASET_CSV, index=False)
    print(f"Wrote tabular dataset to {OUTPUT_DATASET_CSV} ({len(dataset_df)} rows).")

    # Save sequences
    sequences_array = np.stack(sequence_tensors, axis=0)  # Shape (51, 30, 7)
    np.savez_compressed(
        OUTPUT_SEQUENCES_NPZ,
        sequences=sequences_array,
        trip_keys=np.array(trip_keys),
        feature_names=np.array(SEQUENCE_FEATURE_NAMES),
    )
    print(f"Wrote sequence array to {OUTPUT_SEQUENCES_NPZ} shape {sequences_array.shape}.")

    # Generate split manifest
    print("Step 4: Creating split manifest...")
    split_manifest = create_split_manifest(eligible, seed=42)
    OUTPUT_SPLIT_MANIFEST.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    print(f"Wrote split manifest to {OUTPUT_SPLIT_MANIFEST}.")

    # Generate quality report
    print("Step 5: Writing quality report...")
    quality_report = {
        "source": "VED pure-EV telemetry",
        "eligible_trips_count": len(dataset_df),
        "observation_window_duration_s": WINDOW_DURATION_S,
        "sampling_rate_hz": 1.0,
        "sequence_timesteps": WINDOW_STEPS,
        "features": {
            "sequential": SEQUENCE_FEATURE_NAMES,
            "tabular": TABULAR_FEATURE_NAMES,
        },
        "target": {
            "name": "total_trip_distance_km",
            "units": "kilometres",
            "derivation": "Trapezoidal integration of speed over valid intervals for full completed trip",
            "mean": float(dataset_df["total_trip_distance_km"].mean()),
            "std": float(dataset_df["total_trip_distance_km"].std()),
            "min": float(dataset_df["total_trip_distance_km"].min()),
            "max": float(dataset_df["total_trip_distance_km"].max()),
        },
        "trip_quality_details": quality_records,
    }
    OUTPUT_QUALITY_REPORT.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    print(f"Wrote quality report to {OUTPUT_QUALITY_REPORT}.")

    # Generate target definition
    target_def = {
        "task_name": "VED Initial-Window EV Trip Distance Prediction",
        "target_variable": "total_trip_distance_km",
        "target_units": "kilometres",
        "prediction_time": "At t = 30.0 seconds from vehicle trip commencement",
        "observation_window": {
            "duration_s": 30.0,
            "sampling_frequency_hz": 1.0,
            "steps": 30,
        },
        "allowed_inputs": (
            "Telemetry recorded strictly between t = 0 and t = 30 seconds: vehicle speed, "
            "battery current, battery voltage, power, SOC, outside air temperature, and cumulative "
            "window distance."
        ),
        "excluded_inputs": (
            "Future telemetry (t > 30s), final trip duration, final energy consumption, "
            "final SOC, destination GPS coordinates, and any feature derived from completed trip."
        ),
        "distinction": (
            "This task predicts the TOTAL completed trip distance in km, based on initial driving "
            "patterns. It is NOT remaining driving range and must not be presented as battery state of health."
        ),
        "limitations": [
            "Dataset includes only 3 pure-EV vehicles, all 2013 Nissan Leafs.",
            "Vehicle 541 has only 1 eligible trip; vehicle 10 has 12; vehicle 455 has 38.",
            "Generalization across make/model or different climates cannot be assessed.",
            "Sample size of 51 trips is exploratory and academic in nature.",
        ],
    }
    OUTPUT_TARGET_DEFINITION.write_text(json.dumps(target_def, indent=2), encoding="utf-8")
    print(f"Wrote target definition to {OUTPUT_TARGET_DEFINITION}.")
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
