"""Stream VED 7z archives into per-trip summaries for the pure EVs only.

Run from the project root:
    py scripts\\prepare_ved_ev.py

The source archives are read through the system ``tar`` command and are not
expanded onto disk. Trip energy and speed-integrated distance are derived
quantities, not supplied labels; the report preserves data-quality limitations.
"""

from __future__ import annotations

import csv
import json
import math
import posixpath
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "VED"
PROCESSED_DIR = ROOT / "data" / "processed" / "ved"
STATIC_FILE = RAW_DIR / "VED_Static_Data_PHEV_EV.xlsx"
ARCHIVES = [
    RAW_DIR / "VED_DynamicData_Part1.7z",
    RAW_DIR / "VED_DynamicData_Part2.7z",
]
TRIPS_CSV = PROCESSED_DIR / "ved_ev_trip_measurements.csv"
QUALITY_REPORT = PROCESSED_DIR / "ved_ev_quality_report.json"
MANIFEST = ROOT / "data" / "dataset_manifest.json"
MAX_INTERPOLATION_GAP_S = 1.0

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "package_rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

REQUIRED_FIELDS = [
    "DayNum",
    "VehId",
    "Trip",
    "Timestamp(ms)",
    "Latitude[deg]",
    "Longitude[deg]",
    "Vehicle Speed[km/h]",
    "OAT[DegC]",
    "HV Battery Current[A]",
    "HV Battery SOC[%]",
    "HV Battery Voltage[V]",
]

OUTPUT_FIELDS = [
    "source_dataset",
    "source_file",
    "vehicle_id",
    "trip_id",
    "point_count",
    "trip_duration_s",
    "speed_integrated_distance_km",
    "speed_distance_coverage_fraction",
    "gps_path_distance_km",
    "gps_distance_coverage_fraction",
    "battery_signed_energy_kwh",
    "energy_coverage_fraction",
    "energy_consumed_kwh",
    "soc_start_percent",
    "soc_end_percent",
    "soc_delta_percentage_points",
    "outside_air_temperature_mean_c",
    "max_observation_gap_s",
    "gap_interval_count",
    "duplicate_timestamp_count",
    "source_timestamp_order_violation_count",
    "quality_flags",
]


def finite_number(value):
    """Parse a finite number or return None for blank, NaN, or invalid input."""
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def normalize_vehicle_id(value):
    number = finite_number(value)
    if number is None:
        return str(value).strip()
    return str(int(number)) if number.is_integer() else str(number)


def column_index(reference):
    letters = "".join(character for character in reference if character.isalpha())
    index = 0
    for character in letters.upper():
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def read_ved_ev_ids(path):
    """Read pure-EV vehicle IDs from the source XLSX without extra packages."""
    with zipfile.ZipFile(path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_xml = ElementTree.fromstring(
                workbook.read("xl/sharedStrings.xml")
            )
            shared_strings = [
                "".join(text.text or "" for text in item.findall(".//main:t", NS))
                for item in shared_xml.findall("main:si", NS)
            ]

        workbook_xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
        relationships_xml = ElementTree.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        relationships = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships_xml.findall(
                "package_rel:Relationship", NS
            )
        }
        sheets = workbook_xml.findall("main:sheets/main:sheet", NS)
        if not sheets:
            raise ValueError(f"No worksheets found in {path}")

        worksheet_target = relationships[
            sheets[0].attrib[f"{{{NS['office_rel']}}}id"]
        ]
        if worksheet_target.startswith("/"):
            worksheet_name = worksheet_target.lstrip("/")
        else:
            worksheet_name = posixpath.normpath(
                posixpath.join("xl", worksheet_target)
            )
        worksheet_xml = ElementTree.fromstring(workbook.read(worksheet_name))

        rows = worksheet_xml.findall(".//main:sheetData/main:row", NS)
        if not rows:
            raise ValueError(f"No rows found in {path}")

        def row_values(row):
            values = {}
            for cell in row.findall("main:c", NS):
                value = cell.find("main:v", NS)
                if cell.attrib.get("t") == "inlineStr":
                    text = "".join(
                        node.text or ""
                        for node in cell.findall(".//main:t", NS)
                    )
                elif value is None:
                    text = ""
                elif cell.attrib.get("t") == "s":
                    text = shared_strings[int(value.text)]
                else:
                    text = value.text or ""
                values[column_index(cell.attrib["r"])] = text
            return values

        header = row_values(rows[0])
        columns = {value: key for key, value in header.items()}
        if "VehId" not in columns or "EngineType" not in columns:
            raise ValueError(
                f"Expected VehId and EngineType columns in {path}"
            )

        ev_ids = set()
        for row in rows[1:]:
            values = row_values(row)
            if values.get(columns["EngineType"], "").strip().upper() == "EV":
                ev_ids.add(
                    normalize_vehicle_id(values.get(columns["VehId"], ""))
                )
        if not ev_ids:
            raise ValueError(f"No pure EV vehicle IDs found in {path}")
        return ev_ids


def source_members(tar_executable, archive):
    result = subprocess.run(
        [tar_executable, "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [
        member
        for member in result.stdout.splitlines()
        if member.lower().endswith(".csv")
    ]


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    angle = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(min(1.0, math.sqrt(angle)))


def summarize_trip(vehicle_id, trip_id, source_file, rows):
    """Calculate explicit, auditable telemetry-derived trip fields."""
    raw_timestamps = [
        row["timestamp_ms"] for row in rows
        if row["timestamp_ms"] is not None
    ]
    duplicate_timestamp_count = len(raw_timestamps) - len(set(raw_timestamps))
    source_timestamp_order_violations = sum(
        current <= previous
        for previous, current in zip(raw_timestamps, raw_timestamps[1:])
    )
    rows.sort(
        key=lambda row: (
            row["timestamp_ms"] is None,
            row["timestamp_ms"] if row["timestamp_ms"] is not None else 0.0,
        )
    )
    first_time = next(
        (row["timestamp_ms"] for row in rows if row["timestamp_ms"] is not None),
        None,
    )
    last_time = next(
        (
            row["timestamp_ms"]
            for row in reversed(rows)
            if row["timestamp_ms"] is not None
        ),
        None,
    )
    duration_s = (
        max(0.0, (last_time - first_time) / 1000.0)
        if first_time is not None and last_time is not None
        else None
    )

    speed_distance_km = 0.0
    speed_covered_s = 0.0
    gps_distance_km = 0.0
    gps_covered_s = 0.0
    signed_energy_kwh = 0.0
    energy_covered_s = 0.0
    gap_count = 0
    max_gap_s = 0.0
    soc_values = [
        row["soc_percent"] for row in rows if row["soc_percent"] is not None
    ]
    temperature_values = [
        row["outside_air_temperature_c"]
        for row in rows
        if row["outside_air_temperature_c"] is not None
    ]

    for left, right in zip(rows, rows[1:]):
        if left["timestamp_ms"] is None or right["timestamp_ms"] is None:
            continue
        delta_s = (right["timestamp_ms"] - left["timestamp_ms"]) / 1000.0
        if delta_s <= 0:
            gap_count += 1
            continue
        max_gap_s = max(max_gap_s, delta_s)
        if delta_s > MAX_INTERPOLATION_GAP_S:
            gap_count += 1
            continue

        left_speed = left["speed_kmh"]
        right_speed = right["speed_kmh"]
        if left_speed is not None and right_speed is not None:
            speed_distance_km += (
                (left_speed + right_speed) * 0.5 * delta_s / 3600.0
            )
            speed_covered_s += delta_s

        coordinate_values = (
            left["latitude"],
            left["longitude"],
            right["latitude"],
            right["longitude"],
        )
        if all(value is not None for value in coordinate_values):
            gps_distance_km += haversine_km(*coordinate_values)
            gps_covered_s += delta_s

        power_values = (
            left["battery_current_a"],
            right["battery_current_a"],
            left["battery_voltage_v"],
            right["battery_voltage_v"],
        )
        if all(value is not None for value in power_values):
            left_power = left["battery_current_a"] * left["battery_voltage_v"]
            right_power = right["battery_current_a"] * right["battery_voltage_v"]
            signed_energy_kwh += (
                (left_power + right_power) * 0.5 * delta_s / 3_600_000.0
            )
            energy_covered_s += delta_s

    coverage = lambda seconds: (
        min(1.0, seconds / duration_s) if duration_s and duration_s > 0 else 0.0
    )
    soc_start = soc_values[0] if soc_values else None
    soc_end = soc_values[-1] if soc_values else None
    soc_delta = (
        soc_end - soc_start
        if soc_start is not None and soc_end is not None
        else None
    )
    energy_consumed = None
    quality_flags = []
    if coverage(energy_covered_s) < 0.95:
        quality_flags.append("energy_coverage_below_95_percent")
    if coverage(speed_covered_s) < 0.95:
        quality_flags.append("speed_distance_coverage_below_95_percent")
    if gap_count:
        quality_flags.append("nonpositive_or_over_1s_sample_gaps")
    if duplicate_timestamp_count:
        quality_flags.append("duplicate_timestamps")
    if source_timestamp_order_violations:
        quality_flags.append("source_timestamps_not_strictly_increasing")
    if soc_delta is None or abs(soc_delta) < 0.05:
        quality_flags.append("soc_change_unavailable_or_below_0.05_percentage_points")
    elif soc_delta < 0:
        if signed_energy_kwh < 0 and coverage(energy_covered_s) >= 0.95:
            energy_consumed = -signed_energy_kwh
        elif signed_energy_kwh >= 0:
            quality_flags.append("battery_energy_sign_disagrees_with_soc_change")
    elif signed_energy_kwh < 0:
        quality_flags.append("battery_energy_sign_disagrees_with_soc_change")
    else:
        quality_flags.append("net_soc_did_not_decrease_for_consumption_label")

    if speed_distance_km <= 0:
        quality_flags.append("nonpositive_speed_integrated_distance")

    return {
        "source_dataset": "Vehicle Energy Dataset",
        "source_file": source_file,
        "vehicle_id": vehicle_id,
        "trip_id": trip_id,
        "point_count": len(rows),
        "trip_duration_s": duration_s,
        "speed_integrated_distance_km": speed_distance_km,
        "speed_distance_coverage_fraction": coverage(speed_covered_s),
        "gps_path_distance_km": gps_distance_km,
        "gps_distance_coverage_fraction": coverage(gps_covered_s),
        "battery_signed_energy_kwh": signed_energy_kwh,
        "energy_coverage_fraction": coverage(energy_covered_s),
        "energy_consumed_kwh": energy_consumed,
        "soc_start_percent": soc_start,
        "soc_end_percent": soc_end,
        "soc_delta_percentage_points": soc_delta,
        "outside_air_temperature_mean_c": (
            sum(temperature_values) / len(temperature_values)
            if temperature_values
            else None
        ),
        "max_observation_gap_s": max_gap_s,
        "gap_interval_count": gap_count,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "source_timestamp_order_violation_count": (
            source_timestamp_order_violations
        ),
        "quality_flags": ";".join(quality_flags),
    }


def parse_raw_row(row):
    return {
        "timestamp_ms": finite_number(row.get("Timestamp(ms)")),
        "latitude": finite_number(row.get("Latitude[deg]")),
        "longitude": finite_number(row.get("Longitude[deg]")),
        "speed_kmh": finite_number(row.get("Vehicle Speed[km/h]")),
        "outside_air_temperature_c": finite_number(row.get("OAT[DegC]")),
        "battery_current_a": finite_number(row.get("HV Battery Current[A]")),
        "soc_percent": finite_number(row.get("HV Battery SOC[%]")),
        "battery_voltage_v": finite_number(row.get("HV Battery Voltage[V]")),
    }


def parse_member(tar_executable, archive, member, ev_ids):
    process = subprocess.Popen(
        [tar_executable, "-xOf", str(archive), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
    )
    grouped = {}
    row_count = 0
    try:
        reader = csv.DictReader(process.stdout)
        missing = sorted(set(REQUIRED_FIELDS) - set(reader.fieldnames or []))
        if missing:
            process.terminate()
            process.wait()
            raise ValueError(
                f"{member} is missing required VED columns: {missing}"
            )
        for row in reader:
            vehicle_id = normalize_vehicle_id(row.get("VehId"))
            if vehicle_id not in ev_ids:
                continue
            trip_id = (row.get("Trip") or "").strip()
            if not trip_id:
                raise ValueError(f"Missing Trip ID in {member} for {vehicle_id}")
            grouped.setdefault((vehicle_id, trip_id), []).append(
                parse_raw_row(row)
            )
            row_count += 1
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"Could not read {member} from {archive}: {stderr.strip()}"
            )
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    return grouped, row_count


def summarize_measurements(records):
    fields = [
        "trip_duration_s",
        "speed_integrated_distance_km",
        "gps_path_distance_km",
        "battery_signed_energy_kwh",
        "energy_consumed_kwh",
        "soc_delta_percentage_points",
        "outside_air_temperature_mean_c",
        "speed_distance_coverage_fraction",
        "gps_distance_coverage_fraction",
        "energy_coverage_fraction",
    ]
    summaries = {}
    for field in fields:
        values = sorted(
            value for row in records
            if (value := row[field]) is not None and math.isfinite(value)
        )
        missing_count = len(records) - len(values)
        if not values:
            summaries[field] = {
                "missing_count": missing_count,
                "observed_range": {"min": None, "max": None},
                "iqr_outlier_count_review_only": 0,
            }
            continue
        q1 = values[len(values) // 4]
        q3 = values[(3 * len(values)) // 4]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        summaries[field] = {
            "missing_count": missing_count,
            "observed_range": {"min": values[0], "max": values[-1]},
            "iqr_outlier_count_review_only": sum(
                value < lower or value > upper for value in values
            ),
        }
    return summaries


def update_manifest(records, raw_files, quality):
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not update dataset manifest: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("datasets"), list
    ):
        raise ValueError("Dataset manifest must contain a datasets array")

    ved_entries = [
        dataset for dataset in manifest["datasets"]
        if str(dataset.get("source", "")).startswith("Vehicle Energy Dataset")
    ]
    entry = ved_entries[0] if ved_entries else None
    manifest["datasets"] = [
        dataset for dataset in manifest["datasets"]
        if not str(dataset.get("source", "")).startswith("Vehicle Energy Dataset")
        or dataset is entry
    ]
    if entry is None:
        entry = {"source": "Vehicle Energy Dataset"}
        manifest["datasets"].append(entry)
    entry["source"] = "Vehicle Energy Dataset"
    entry.update({
        "source_url": (
            "https://github.com/gsoh/VED/tree/"
            "6baa4963782d515a67d32a5490bd5d11f5d9bf0d"
        ),
        "status": "processed_derived_trip_energy_not_range_not_training_ready",
        "raw_path": "data/raw/VED",
        "raw_files": raw_files,
        "processed_file": "data/processed/ved/ved_ev_trip_measurements.csv",
        "quality_report": "data/processed/ved/ved_ev_quality_report.json",
        "sample_count": len(records),
        "vehicle_count": len(quality["vehicle_ids"]),
        "license": (
            "The source repository states Apache License 2.0; scope over "
            "bundled measurements is not separately described. Acquired for "
            "local inspection only; do not redistribute."
        ),
        "observed_measurements": [
            "GPS latitude/longitude",
            "vehicle speed (km/h)",
            "outside-air temperature (degC)",
            "HV battery current (A)",
            "HV battery voltage (V)",
            "HV battery SOC (%)",
        ],
        "calculated_targets": {
            "energy_consumed_kwh": {
                "definition": (
                    "Negative integral of source HV battery current times "
                    "voltage over valid intervals, retained only when sign "
                    "agrees with a decrease in recorded SOC."
                ),
                "status": "derived_trip_energy_not_vehicle_range",
            },
            "speed_integrated_distance_km": {
                "definition": (
                    "Trapezoidal integration of recorded vehicle speed over "
                    "valid timestamp intervals."
                ),
                "status": "derived_distance_not_odometer_measurement",
            },
        },
        "training_status": (
            "Not ready for range prediction: trip range is not supplied, "
            "energy labels are derived and quality-filtered, and only three "
            "pure EV identities are available."
        ),
        "parser_status": "source_specific_parser_completed",
    })
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    missing = [path for path in [*ARCHIVES, STATIC_FILE, MANIFEST] if not path.exists()]
    if missing:
        sys.exit("Required VED input is missing: " + ", ".join(map(str, missing)))
    tar_executable = shutil.which("tar")
    if not tar_executable:
        sys.exit(
            "The system tar utility is required to stream VED 7z archives. "
            "No archive extraction or metrics were produced."
        )

    ev_ids = read_ved_ev_ids(STATIC_FILE)
    summaries = []
    raw_trip_ids = set()
    source_row_count = 0
    raw_files = [STATIC_FILE.relative_to(ROOT).as_posix()]
    for archive in ARCHIVES:
        members = source_members(tar_executable, archive)
        if not members:
            raise ValueError(f"No CSV members found in {archive}")
        for member in members:
            grouped, row_count = parse_member(
                tar_executable, archive, member, ev_ids
            )
            source_row_count += row_count
            source_file = member.rsplit("/", 1)[-1]
            raw_files.append(archive.relative_to(ROOT).as_posix())
            for (vehicle_id, trip_id), rows in grouped.items():
                key = (vehicle_id, trip_id)
                if key in raw_trip_ids:
                    raise ValueError(
                        "VED declares VehId/Trip unique, but duplicate trip "
                        f"{vehicle_id}/{trip_id} occurs in multiple weekly files. "
                        "Resolve before writing trip labels."
                    )
                raw_trip_ids.add(key)
                summaries.append(
                    summarize_trip(vehicle_id, trip_id, source_file, rows)
                )

    if not summaries:
        raise ValueError("No rows for pure EV IDs were found in the VED archives")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summaries.sort(key=lambda row: (row["vehicle_id"], row["trip_id"]))
    with TRIPS_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    quality_flag_counts = Counter()
    for record in summaries:
        quality_flag_counts.update(
            flag for flag in record["quality_flags"].split(";") if flag
        )
    quality = {
        "source": "Vehicle Energy Dataset (VED)",
        "source_url": (
            "https://github.com/gsoh/VED/tree/"
            "6baa4963782d515a67d32a5490bd5d11f5d9bf0d"
        ),
        "license_note": (
            "Repository states Apache 2.0; license scope over measurements "
            "is not separately described. Local inspection only."
        ),
        "raw_archives_unchanged": True,
        "source_point_rows_for_pure_evs": source_row_count,
        "trip_count": len(summaries),
        "vehicle_ids": sorted(ev_ids, key=lambda value: int(value)),
        "trip_count_by_vehicle": dict(Counter(
            record["vehicle_id"] for record in summaries
        )),
        "energy_labeled_trip_count": sum(
            record["energy_consumed_kwh"] is not None
            for record in summaries
        ),
        "energy_labeled_trip_count_by_vehicle": dict(Counter(
            record["vehicle_id"] for record in summaries
            if record["energy_consumed_kwh"] is not None
        )),
        "measurement_summary_method": (
            "Ranges are observed min/max; outliers are review-only counts "
            "using 1.5*IQR fences with nearest-rank quartiles."
        ),
        "measurement_summaries": summarize_measurements(summaries),
        "quality_flag_counts": dict(quality_flag_counts),
        "timestamp_quality": {
            "trips_with_duplicate_timestamps": sum(
                record["duplicate_timestamp_count"] > 0
                for record in summaries
            ),
            "duplicate_timestamp_count": sum(
                record["duplicate_timestamp_count"] for record in summaries
            ),
            "trips_with_source_order_violations": sum(
                record["source_timestamp_order_violation_count"] > 0
                for record in summaries
            ),
            "source_order_violation_count": sum(
                record["source_timestamp_order_violation_count"]
                for record in summaries
            ),
        },
        "missing_measurements": {
            "energy_consumed_kwh": sum(
                record["energy_consumed_kwh"] is None for record in summaries
            ),
            "speed_integrated_distance_km": sum(
                record["speed_distance_coverage_fraction"] < 0.95
                for record in summaries
            ),
            "gps_path_distance_km": sum(
                record["gps_distance_coverage_fraction"] < 0.95
                for record in summaries
            ),
            "outside_air_temperature_mean_c": sum(
                record["outside_air_temperature_mean_c"] is None
                for record in summaries
            ),
        },
        "target_definitions": {
            "energy_consumed_kwh": (
                "Derived per-trip net battery energy from trapezoidal "
                "integration of HV Battery Current[A] * HV Battery Voltage[V], "
                "with sign retained only when consistent with decreasing SOC."
            ),
            "speed_integrated_distance_km": (
                "Derived per-trip distance from trapezoidal integration of "
                "Vehicle Speed[km/h] against Timestamp(ms)."
            ),
            "range_km": "Not supplied; no range labels are claimed.",
        },
        "interval_policy": {
            "maximum_interpolation_gap_s": MAX_INTERPOLATION_GAP_S,
            "larger_or_nonpositive_intervals": (
                "Not integrated; counted and exposed by quality flags."
            ),
        },
        "limitations": [
            "Only three pure EV identities are included; all are 2013 Nissan Leafs.",
            "Trip energy and speed-integrated distance are derived, not supplied labels.",
            "Battery-current sign is checked against SOC change per trip; mismatches are unlabeled.",
            "Trips with under 95% telemetry coverage are not assigned an energy label.",
            "Only outside-air temperature is available; weather-aware performance is not supported.",
            "There is no direct, measured vehicle range target in the source.",
        ],
        "parser_status": "processed",
        "training_status": (
            "Not ready for range prediction; an energy-consumption task may be "
            "considered after reviewing label coverage and identity-aware test size."
        ),
    }
    QUALITY_REPORT.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    update_manifest(
        summaries,
        sorted(set(raw_files)),
        quality,
    )
    print(f"Pure EV IDs: {', '.join(quality['vehicle_ids'])}")
    print(f"Source EV telemetry rows: {source_row_count}")
    print(f"Trip summaries: {len(summaries)}")
    print(f"Trips with sign/coverage-qualified energy: {quality['energy_labeled_trip_count']}")
    print(f"Trip data: {TRIPS_CSV}")
    print(f"Quality report: {QUALITY_REPORT}")


if __name__ == "__main__":
    main()
