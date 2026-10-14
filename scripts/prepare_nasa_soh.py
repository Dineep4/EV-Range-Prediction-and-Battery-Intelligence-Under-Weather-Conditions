
"""Convert NASA PCoE Bxxxx MATLAB battery files into a cycle-level SOH table.

Run from the project root:
    python scripts/prepare_nasa_soh.py

SOH definition:
    100 * measured discharge capacity / median of the first 10
    valid positive discharge capacities for that battery.

This is a provisional within-battery relative reference, not the
manufacturer-rated capacity or a validated absolute SOH measurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "NASA"
OUT_DIR = ROOT / "data" / "processed"
OUT_CSV = OUT_DIR / "battery_measurements.csv"
REPORT = OUT_DIR / "nasa_preparation_report.json"
QUALITY_REPORT = OUT_DIR / "nasa_quality_report.json"
MANIFEST = ROOT / "data" / "dataset_manifest.json"


def field(obj, name, default=None):
    """Read a field from a MATLAB structure."""
    if obj is None:
        return default

    if hasattr(obj, name):
        return getattr(obj, name)

    if isinstance(obj, np.void) and obj.dtype.names:
        if name in obj.dtype.names:
            value = obj[name]
            if isinstance(value, np.ndarray) and value.size == 1:
                return value.item()
            return value

    if isinstance(obj, dict):
        return obj.get(name, default)

    return default


def scalar(value):
    """Convert a value to a finite scalar, if possible."""
    if value is None:
        return np.nan

    try:
        arr = np.asarray(value).squeeze()

        if arr.size == 0:
            return np.nan

        result = float(arr.reshape(-1)[0])
        return result if np.isfinite(result) else np.nan

    except (TypeError, ValueError, OverflowError):
        return np.nan


def numeric_array(value):
    """Return finite numeric values from an array."""
    if value is None:
        return np.array([], dtype=float)

    try:
        arr = np.asarray(value, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    except (TypeError, ValueError):
        return np.array([], dtype=float)


def mean_or_nan(value):
    arr = numeric_array(value)
    return float(np.mean(arr)) if arr.size else np.nan


def capacity_review_flags(capacity: float) -> list[str]:
    """Flag unusual positive capacities for review without excluding them."""
    flags = []
    if capacity < 0.5:
        flags.append("below_0.5_ah_review_only")
    if capacity > 2.0:
        flags.append("above_2.0_ah_review_only")
    return flags


def ved_manifest_entry():
    """Keep VED acquisition and parser status when NASA reports are refreshed."""
    ved_dir = ROOT / "data" / "raw" / "VED"
    archive_files = sorted(ved_dir.glob("VED_DynamicData_Part*.7z"))
    static_file = ved_dir / "VED_Static_Data_PHEV_EV.xlsx"
    raw_files = [path for path in [*archive_files, static_file] if path.exists()]
    quality_path = ROOT / "data" / "processed" / "ved" / "ved_ev_quality_report.json"
    processed = quality_path.exists()

    entry = {
        "source": "Vehicle Energy Dataset",
        "source_url": (
            "https://github.com/gsoh/VED/tree/"
            "6baa4963782d515a67d32a5490bd5d11f5d9bf0d"
        ),
        "status": (
            "processed_derived_trip_energy_not_range_not_training_ready"
            if processed
            else "not_acquired_license_scope_unverified"
        ),
        "raw_path": "data/raw/VED",
        "raw_files": [path.relative_to(ROOT).as_posix() for path in raw_files],
        "sample_count": None,
        "documented_formats": [
            "Weekly dynamic CSV files in two 7z archives",
            "Vehicle specifications in XLSX",
        ],
        "documented_fields": [
            "Vehicle and trip IDs",
            "GPS coordinates and speed",
            "Outside-air temperature",
            "High-voltage battery current, voltage, and SOC",
        ],
        "license": (
            "Repository states Apache License 2.0; the scope over the "
            "measurements is not separately described."
        ),
        "target_definition": (
            "Trip energy and distance are derived from telemetry; range is "
            "not supplied as a direct label."
        ),
        "ev_coverage_limitation": (
            "Three pure EVs are reported, all 2013 Nissan Leafs."
        ),
        "parser_status": (
            "source_specific_parser_completed" if processed else "not_implemented"
        ),
    }
    if raw_files:
        entry["status"] = (
            "processed_derived_trip_energy_not_range_not_training_ready"
            if processed
            else "acquired_not_processed_license_scope_unverified"
        )
    if processed:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        entry.update({
            "sample_count": quality["trip_count"],
            "quality_report": "data/processed/ved/ved_ev_quality_report.json",
            "processed_file": "data/processed/ved/ved_ev_trip_measurements.csv",
            "vehicle_count": len(quality["vehicle_ids"]),
            "source_point_rows_for_pure_evs": quality[
                "source_point_rows_for_pure_evs"
            ],
            "energy_labeled_trip_count": quality["energy_labeled_trip_count"],
            "training_status": quality["training_status"],
        })
    else:
        entry["training_status"] = (
            "Not acquired or processed; verify license scope and source files first."
        )
    return entry


def discharge_trace_diagnostics(data):
    """Summarize raw trace consistency without treating it as a capacity label."""
    voltage = numeric_array(field(data, "Voltage_measured"))
    current_value = field(data, "Current_measured")
    time_value = field(data, "Time")
    current = numeric_array(current_value)
    time_values = numeric_array(time_value)

    current_integral = None
    if current_value is not None and time_value is not None:
        try:
            raw_current = np.asarray(current_value, dtype=float).ravel()
            raw_time = np.asarray(time_value, dtype=float).ravel()
            if raw_current.size == raw_time.size:
                paired_finite = np.isfinite(raw_current) & np.isfinite(raw_time)
                paired_current = raw_current[paired_finite]
                paired_time = raw_time[paired_finite]
                if paired_time.size > 1:
                    order = np.argsort(paired_time, kind="stable")
                    ordered_time = paired_time[order]
                    ordered_current = np.abs(paired_current[order])
                    current_integral = float(
                        np.sum(
                            (ordered_current[1:] + ordered_current[:-1])
                            * np.diff(ordered_time)
                            * 0.5
                        ) / 3600.0
                    )
        except (TypeError, ValueError, OverflowError):
            current_integral = None

    return {
        "voltage_min_v": float(np.min(voltage)) if voltage.size else None,
        "voltage_last_v": float(voltage[-1]) if voltage.size else None,
        "duration_s": (
            float(np.max(time_values) - np.min(time_values))
            if time_values.size > 1
            else None
        ),
        "voltage_sample_count": int(voltage.size),
        "current_sample_count": int(current.size),
        "time_sample_count": int(time_values.size),
        "absolute_current_integral_ah_diagnostic": current_integral,
    }


def write_dataset_manifest(files, frame, reports):
    excluded_count = sum(
        len(report.get("excluded_rows", [])) for report in reports
    )
    raw_discharge_count = sum(
        report.get("discharge_cycles_seen", 0) for report in reports
    )
    review_count = sum(
        len(report.get("review_rows", [])) for report in reports
    )

    manifest = {
        "schema_version": "1.0",
        "datasets": [
            {
                "source": "NASA PCoE Li-ion Battery Aging Dataset",
                "source_url": "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets",
                "status": "processed_not_training_ready",
                "raw_path": "data/raw/NASA",
                "raw_files": [
                    path.relative_to(ROOT).as_posix() for path in files
                ],
                "raw_file_count": len(files),
                "processed_file": "data/processed/battery_measurements.csv",
                "quality_report": "data/processed/nasa_quality_report.json",
                "sample_count": int(len(frame)),
                "raw_discharge_cycles": raw_discharge_count,
                "excluded_observation_count": excluded_count,
                "review_flagged_row_count": int(review_count),
                "battery_count": int(frame["battery_id"].nunique()),
                "observed_measurements": {
                    "capacity_ah": {
                        "units": "Ah",
                        "definition": "NASA discharge-cycle capacity field",
                    },
                    "voltage_mean": {"units": "V"},
                    "current_mean": {"units": "A"},
                    "temperature_mean": {"units": "degC"},
                    "ambient_temperature_c": {"units": "degC"},
                },
                "calculated_targets": {
                    "soh_percent": {
                        "units": "%",
                        "definition": (
                            "100 * discharge capacity / median of the first "
                            "10 valid positive discharge capacities per battery"
                        ),
                        "status": "provisional_not_ready_for_benchmark_training",
                    },
                },
                "preprocessing_status": (
                    "Positive unusual capacities are retained and flagged. "
                    "Missing, non-finite, and non-positive capacities are "
                    "excluded with source cycle and reason in the preparation report. "
                    "Capacity review thresholds are non-scientific triage flags."
                ),
                "training_status": (
                    "Blocked: current low-capacity observations and target "
                    "reference are unresolved; protocols differ across batteries."
                ),
            },
            {
                "source": "CALCE CS2 battery cycling data",
                "source_url": "https://calce.umd.edu/data",
                "status": "official_source_verified_not_acquired_license_unspecified",
                "sample_count": None,
                "documented_formats": [
                    "Arbin cell test files: Excel collections",
                    "CADEX files for CS2_8 and CS2_21: TXT",
                ],
                "candidate_data": (
                    "Official page documents 1.1 Ah CS2 cells, CC/CV charging, "
                    "usually 2.7 V discharge cut-off, multiple cycling profiles, "
                    "and file collections. No specific data license is stated; "
                    "actual capacity channels, units, and archive sizes remain "
                    "unverified until authorized files are inspected."
                ),
                "terms": (
                    "CALCE describes the data as open access and requests "
                    "citation for publications; the official page does not "
                    "identify a dataset-specific license."
                ),
                "parser_status": "not_implemented",
            },
            {
                "source": "Oxford Battery Degradation Dataset 1",
                "source_url": (
                    "https://ora.ox.ac.uk/objects/uuid:"
                    "03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac"
                ),
                "status": "official_record_verified_not_acquired_license_and_payload_unverified",
                "doi": "10.5287/bodleian:KO2kdmYGg",
                "sample_count": None,
                "documented_population": (
                    "Eight Kokam SLPB533459H4 740 mAh lithium-ion pouch cells"
                ),
                "documented_format": (
                    "The official record refers to readme.txt; the payload "
                    "file formats and measured channels could not be verified "
                    "from the public metadata."
                ),
                "terms": (
                    "No explicit reuse license or dataset terms were visible "
                    "in the official record metadata reviewed."
                ),
                "parser_status": "not_implemented",
            },
            ved_manifest_entry(),
        ],
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def write_quality_report(frame, reports):
    numeric_columns = {
        "capacity_ah": "Ah",
        "reference_capacity_ah": "Ah",
        "soh_percent": "%",
        "ambient_temperature_c": "degC",
        "voltage_mean": "V",
        "voltage_min": "V",
        "voltage_max": "V",
        "current_mean": "A",
        "current_abs_mean": "A",
        "temperature_mean": "degC",
        "temperature_max": "degC",
        "discharge_duration_s": "s",
    }
    measurements = {}
    for column, units in numeric_columns.items():
        values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            outliers = {"method": "1.5 * IQR fences", "count": 0}
            value_range = {"min": None, "max": None}
        else:
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = {
                "method": "1.5 * IQR fences; review flag only, no removal",
                "lower_fence": lower,
                "upper_fence": upper,
                "count": int(((values < lower) | (values > upper)).sum()),
            }
            value_range = {
                "min": float(values.min()),
                "max": float(values.max()),
            }
        measurements[column] = {
            "units": units,
            "missing_count": int(frame[column].isna().sum()),
            "observed_range": value_range,
            "outliers": outliers,
        }

    ordered = frame.sort_values(["battery_id", "cycle_index"])
    cycle_gaps = int(
        ordered.groupby("battery_id")["cycle_index"].diff().gt(1).sum()
    )
    excluded_rows = [
        row
        for report in reports
        for row in report.get("excluded_rows", [])
    ]
    review_count = int(frame["capacity_review_flags"].fillna("").ne("").sum())
    review_flag_counts = {
        "below_0.5_ah_retained": int(frame["capacity_ah"].lt(0.5).sum()),
        "above_2.0_ah_retained": int(frame["capacity_ah"].gt(2.0).sum()),
    }
    quality_report = {
        "source": "NASA PCoE Li-ion Battery Aging Dataset",
        "processed_file": "data/processed/battery_measurements.csv",
        "sample_count": int(len(frame)),
        "battery_count": int(frame["battery_id"].nunique()),
        "measurements": measurements,
        "duplicates": {
            "key": ["battery_id", "cycle_index"],
            "count": int(frame.duplicated(["battery_id", "cycle_index"]).sum()),
        },
        "discharge_cycle_gaps_after_exclusions": cycle_gaps,
        "positive_capacity_review_rows_retained": review_count,
        "capacity_review_flag_counts": review_flag_counts,
        "review_thresholds_are_non_scientific_triage_only": True,
        "trace_diagnostic_note": (
            "The per-row preparation report records raw trace minima, "
            "sample counts, duration, and sparse absolute-current integral "
            "for flagged and excluded observations. The integral is not a "
            "capacity label or an exclusion criterion."
        ),
        "excluded_observations": {
            "count": len(excluded_rows),
            "rows": excluded_rows,
        },
        "soh_target": {
            "definition": (
                "100 * measured discharge capacity / median of the first "
                "10 valid positive discharge capacities per battery"
            ),
            "status": "provisional_not_ready_for_benchmark_training",
            "observed_range_percent": {
                "min": float(frame["soh_percent"].min()),
                "max": float(frame["soh_percent"].max()),
            },
            "above_100_percent_count": int(
                (frame["soh_percent"] > 100).sum()
            ),
            "above_120_percent_count": int(
                (frame["soh_percent"] > 120).sum()
            ),
            "below_50_percent_count": int(
                (frame["soh_percent"] < 50).sum()
            ),
            "reason_for_review": (
                "The first-10-capacity reference turns some documented very "
                "low readings into implausibly high ratios. The source itself "
                "says B0041-B0044 contain unexplained low-capacity discharges; "
                "different cells also use different temperatures, currents, "
                "profiles, and voltage cut-offs."
            ),
        },
    }
    QUALITY_REPORT.write_text(
        json.dumps(quality_report, indent=2),
        encoding="utf-8",
    )


def first_mat_struct(mat):
    """Find the top-level MATLAB battery structure."""
    keys = [key for key in mat.keys() if not key.startswith("__")]

    if not keys:
        return None, None

    for key in keys:
        obj = mat[key]
        if field(obj, "cycle") is not None:
            return key, obj

    return keys[0], mat[keys[0]]


def as_items(value):
    """Convert MATLAB cycle collections into a Python list."""
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return list(value.ravel())

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def convert_file(path: Path):
    """Extract discharge measurements and calculate provisional SOH."""
    mat = loadmat(
        str(path),
        squeeze_me=True,
        struct_as_record=False,
    )

    top_key, battery = first_mat_struct(mat)
    cycles = field(battery, "cycle")

    if cycles is None:
        raise ValueError(
            f"No cycle field found in top-level variable {top_key!r}"
        )

    battery_id = str(top_key or path.stem)
    discharge_rows = []
    excluded_rows = []
    review_rows = []
    discharge_number = 0
    discharge_cycles_seen = 0

    for original_index, cycle in enumerate(as_items(cycles), start=1):
        kind = field(cycle, "type", "")

        if isinstance(kind, bytes):
            kind = kind.decode("utf-8", errors="replace")

        kind = str(kind).strip().lower()

        if kind != "discharge":
            continue

        discharge_cycles_seen += 1
        discharge_number += 1
        data = field(cycle, "data")
        capacity = scalar(field(data, "Capacity"))

        trace = discharge_trace_diagnostics(data)

        if not np.isfinite(capacity):
            excluded_rows.append({
                "battery_id": battery_id,
                "source_dataset": "NASA PCoE Li-ion Battery Aging Dataset",
                "source_file": path.name,
                "source_cycle_index": original_index,
                "reason": "missing_or_non_finite_capacity",
                **trace,
            })
            continue

        if capacity <= 0:
            excluded_rows.append({
                "battery_id": battery_id,
                "source_dataset": "NASA PCoE Li-ion Battery Aging Dataset",
                "source_file": path.name,
                "source_cycle_index": original_index,
                "capacity_ah": capacity,
                "reason": "non_positive_capacity",
                **trace,
            })
            continue

        review_flags = capacity_review_flags(capacity)
        if review_flags:
            review_rows.append({
                "battery_id": battery_id,
                "source_file": path.name,
                "source_cycle_index": original_index,
                "capacity_ah": capacity,
                "flags": review_flags,
                **trace,
            })

        voltage = numeric_array(field(data, "Voltage_measured"))
        current = numeric_array(field(data, "Current_measured"))
        temperature = numeric_array(
            field(data, "Temperature_measured")
        )
        time_values = numeric_array(field(data, "Time"))

        ambient = scalar(field(cycle, "ambient_temperature"))

        duration = (
            float(np.max(time_values) - np.min(time_values))
            if time_values.size > 1
            else np.nan
        )

        discharge_rows.append({
            "battery_id": battery_id,
            "source_dataset": "NASA PCoE Li-ion Battery Aging Dataset",
            "source_file": path.name,
            "cycle_index": discharge_number,
            "source_cycle_index": original_index,
            "capacity_ah": capacity,
            "capacity_review_flags": ";".join(review_flags),
            "ambient_temperature_c": ambient,
            "voltage_mean": (
                float(np.mean(voltage)) if voltage.size else np.nan
            ),
            "voltage_min": (
                float(np.min(voltage)) if voltage.size else np.nan
            ),
            "voltage_max": (
                float(np.max(voltage)) if voltage.size else np.nan
            ),
            "current_mean": (
                float(np.mean(current)) if current.size else np.nan
            ),
            "current_abs_mean": (
                float(np.mean(np.abs(current)))
                if current.size else np.nan
            ),
            "temperature_mean": (
                float(np.mean(temperature))
                if temperature.size else np.nan
            ),
            "temperature_max": (
                float(np.max(temperature))
                if temperature.size else np.nan
            ),
            "discharge_duration_s": duration,
        })

    if not discharge_rows:
        return [], {
            "file": path.name,
            "battery_id": battery_id,
            "status": "no_valid_discharge_capacity",
            "discharge_cycles_seen": discharge_cycles_seen,
            "retained_rows": 0,
            "excluded_rows": excluded_rows,
            "review_rows": review_rows,
        }

    frame = pd.DataFrame(discharge_rows)
    frame = frame.sort_values("cycle_index").reset_index(drop=True)

    # Provisional reference: median of the first 10 valid positive
    # discharge capacities. This reduces sensitivity to one unusual
    # first measurement, but does not prove those cycles are healthy.
    early_capacities = frame["capacity_ah"].head(10)
    baseline = float(early_capacities.median())

    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError(
            f"Invalid reference capacity for {battery_id}"
        )

    frame["reference_capacity_ah"] = baseline
    frame["soh_percent"] = (
        100.0 * frame["capacity_ah"] / baseline
    )
    frame["soh_definition"] = (
        "100 * measured discharge capacity / median of first 10 "
        "valid positive discharge capacities; provisional reference"
    )

    report = {
        "file": path.name,
        "battery_id": battery_id,
        "status": "ok",
        "discharge_cycles_seen": discharge_cycles_seen,
        "valid_discharge_rows": int(len(frame)),
        "retained_rows": int(len(frame)),
        "excluded_rows": excluded_rows,
        "review_rows": review_rows,
        "capacity_review_thresholds_ah": {
            "below": 0.5,
            "above": 2.0,
            "action": (
                "non-scientific review flags only; all finite positive "
                "capacities are retained"
            ),
            "review_flag_counts": {
                "below_0.5_ah": int(
                    frame["capacity_ah"].lt(0.5).sum()
                ),
                "above_2.0_ah": int(
                    frame["capacity_ah"].gt(2.0).sum()
                ),
            },
        },
        "review_interpretation": (
            "Review rows include trace diagnostics. The absolute current "
            "integral is a sparse-sample consistency diagnostic only; it is "
            "not a replacement capacity measurement."
        ),
        "reference_capacity_ah": baseline,
        "capacity_min_ah": float(frame["capacity_ah"].min()),
        "capacity_max_ah": float(frame["capacity_ah"].max()),
        "soh_min_percent": float(frame["soh_percent"].min()),
        "soh_max_percent": float(frame["soh_percent"].max()),
        "soh_reference_method": (
            "Median of first 10 valid positive discharge capacities; "
            "provisional and requires scientific validation"
        ),
        "soh_label_status": "provisional_not_ready_for_benchmark_training",
    }

    return frame.to_dict(orient="records"), report


def main():
    if not RAW_DIR.exists():
        sys.exit(f"NASA raw-data folder not found: {RAW_DIR}")

    files = sorted(RAW_DIR.rglob("*.mat"))

    if not files:
        sys.exit(f"No .mat files found under {RAW_DIR}")

    all_rows = []
    reports = []

    for path in files:
        try:
            rows, report = convert_file(path)
            all_rows.extend(rows)
            reports.append(report)

            row_count = report.get(
                "valid_discharge_rows",
                report.get("retained_rows", 0),
            )

            print(
                f"{path.name}: {report['status']} — "
                f"{row_count} retained discharge rows; "
                f"{len(report.get('excluded_rows', []))} excluded with reasons; "
                f"{len(report.get('review_rows', []))} flagged for review"
            )

        except Exception as exc:
            reports.append({
                "file": path.name,
                "status": "error",
                "error": str(exc),
            })
            print(f"{path.name}: ERROR — {exc}")

    if not all_rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        REPORT.write_text(
            json.dumps(reports, indent=2),
            encoding="utf-8",
        )

        sys.exit(f"No usable rows were extracted. See {REPORT}")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(
        ["battery_id", "cycle_index"]
    ).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_CSV, index=False)

    REPORT.write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )
    write_quality_report(df, reports)
    write_dataset_manifest(files, df, reports)

    print("\nConversion complete.")
    print(f"Batteries: {df['battery_id'].nunique()}")
    print(f"Rows: {len(df)}")
    print(f"CSV: {OUT_CSV}")
    print(f"Report: {REPORT}")
    print(f"Quality report: {QUALITY_REPORT}")
    print(f"Dataset manifest: {MANIFEST}")
    print(
        "SOH reference: median of the first 10 valid positive "
        "discharge capacities per battery (provisional)."
    )
    print(
        "Inspect the resulting SOH distributions before model training."
    )


if __name__ == "__main__":
    main()
