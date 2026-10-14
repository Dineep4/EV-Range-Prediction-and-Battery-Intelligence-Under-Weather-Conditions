"""Read-only audit of NASA battery capacity and SOH-reference evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from scripts.prepare_nasa_soh import (
    capacity_review_flags,
    discharge_trace_diagnostics,
    field,
    scalar,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "NASA"
PROCESSED_CSV = ROOT / "data" / "processed" / "battery_measurements.csv"
PREPARATION_REPORT = ROOT / "data" / "processed" / "nasa_preparation_report.json"
QUALITY_REPORT = ROOT / "data" / "processed" / "nasa_quality_report.json"
OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "nasa_soh_reference_evidence_audit.json"
)
NASA_SOURCE_URL = "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets"

PROTOCOL_GROUPS: list[dict[str, Any]] = [
    {
        "readme": "README1.txt",
        "batteries": ["B0005", "B0006", "B0007", "B0018"],
        "ambient_temperature_c": 24,
        "load_profile": "2 A constant-current discharge",
        "cutoff_voltage_v": {
            "B0005": 2.7,
            "B0006": 2.5,
            "B0007": 2.2,
            "B0018": 2.5,
        },
        "readme_eol_capacity_ah": 1.4,
        "readme_eol_fade_percent": 30,
        "readme_rated_capacity_ah": 2.0,
    },
    {
        "readme": "README_25_26_27_28.txt",
        "batteries": ["B0025", "B0026", "B0027", "B0028"],
        "ambient_temperature_c": 24,
        "load_profile": "0.05 Hz square wave, 4 A amplitude, 50% duty",
        "cutoff_voltage_v": {
            "B0025": 2.0,
            "B0026": 2.2,
            "B0027": 2.5,
            "B0028": 2.7,
        },
        "readme_eol_capacity_ah": None,
        "readme_eol_fade_percent": None,
        "readme_rated_capacity_ah": None,
    },
    {
        "readme": "README_29_30_31_32.txt",
        "batteries": ["B0029", "B0030", "B0031", "B0032"],
        "ambient_temperature_c": 43,
        "load_profile": "4 A discharge",
        "cutoff_voltage_v": {
            "B0029": 2.0,
            "B0030": 2.2,
            "B0031": 2.5,
            "B0032": 2.7,
        },
        "readme_eol_capacity_ah": None,
        "readme_eol_fade_percent": None,
        "readme_rated_capacity_ah": None,
    },
    {
        "readme": "README_33_34_36.txt",
        "batteries": ["B0033", "B0034", "B0036"],
        "ambient_temperature_c": 24,
        "load_profile": {
            "B0033": "4 A discharge",
            "B0034": "4 A discharge",
            "B0036": "2 A discharge",
        },
        "cutoff_voltage_v": {
            "B0033": 2.0,
            "B0034": 2.2,
            "B0036": 2.7,
        },
        "readme_eol_capacity_ah": 1.6,
        "readme_eol_fade_percent": 20,
        "readme_rated_capacity_ah": None,
    },
    {
        "readme": "README_38_39_40.txt",
        "batteries": ["B0038", "B0039", "B0040"],
        "ambient_temperature_c": [24, 44],
        "load_profile": "Multiple discharge loads: 1 A, 2 A, and 4 A",
        "cutoff_voltage_v": {
            "B0038": 2.2,
            "B0039": 2.5,
            "B0040": 2.7,
        },
        "readme_eol_capacity_ah": 1.6,
        "readme_eol_fade_percent": 20,
        "readme_rated_capacity_ah": None,
    },
    {
        "readme": "README_41_42_43_44.txt",
        "batteries": ["B0041", "B0042", "B0043", "B0044"],
        "ambient_temperature_c": 4,
        "load_profile": "Multiple fixed loads: 1 A and 4 A",
        "cutoff_voltage_v": {
            "B0041": 2.0,
            "B0042": 2.2,
            "B0043": 2.5,
            "B0044": 2.7,
        },
        "readme_eol_capacity_ah": 1.4,
        "readme_eol_fade_percent": 30,
        "readme_rated_capacity_ah": None,
        "readme_unresolved_capacity_note": (
            "Several discharge runs have very low capacity; reasons have "
            "not been fully analyzed."
        ),
    },
]


def items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def matlab_time_components(value: Any) -> list[float] | None:
    """Preserve a MATLAB date vector without guessing a timezone."""
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if values.size != 6 or not np.isfinite(values).all():
        return None
    return [float(component) for component in values]


def summarize_mat_file(path: Path) -> dict[str, Any]:
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    keys = [key for key in mat if not key.startswith("__")]
    if not keys:
        raise ValueError(f"No MATLAB data variables found in {path.name}.")
    battery = next(
        (mat[key] for key in keys if field(mat[key], "cycle") is not None),
        None,
    )
    if battery is None:
        raise ValueError(f"No cycle structure found in {path.name}.")

    cycles = items(field(battery, "cycle"))
    operation_counts: dict[str, int] = {}
    discharge_records: list[dict[str, Any]] = []
    for source_cycle_index, cycle in enumerate(cycles, start=1):
        operation = field(cycle, "type", "")
        if isinstance(operation, bytes):
            operation = operation.decode(errors="replace")
        operation = str(operation).strip().lower()
        operation_counts[operation or "missing"] = (
            operation_counts.get(operation or "missing", 0) + 1
        )
        if operation != "discharge":
            continue

        data = field(cycle, "data")
        capacity = scalar(field(data, "Capacity"))
        diagnostics = discharge_trace_diagnostics(data)
        flags = (
            capacity_review_flags(capacity)
            if np.isfinite(capacity) and capacity > 0
            else []
        )
        discharge_records.append({
            "source_file": path.name,
            "battery_id": path.stem,
            "source_cycle_index": source_cycle_index,
            "operation": operation,
            "start_time_components_matlab_date_vector": matlab_time_components(
                field(cycle, "time")
            ),
            "ambient_temperature_c": scalar(
                field(cycle, "ambient_temperature")
            ),
            "capacity_ah": capacity if np.isfinite(capacity) else None,
            "capacity_review_flags": flags,
            **diagnostics,
        })

    positive = [
        record
        for record in discharge_records
        if record["capacity_ah"] is not None
        and record["capacity_ah"] > 0
    ]
    values = np.asarray(
        [record["capacity_ah"] for record in positive],
        dtype=float,
    )
    ordered_positive = sorted(
        positive, key=lambda record: record["source_cycle_index"]
    )
    current_reference = (
        float(np.median([
            record["capacity_ah"] for record in ordered_positive[:10]
        ]))
        if len(ordered_positive) >= 10
        else None
    )
    processed = pd.read_csv(PROCESSED_CSV)
    processed_battery = processed.loc[processed["battery_id"] == path.stem]
    recorded_reference = (
        float(processed_battery["reference_capacity_ah"].iloc[0])
        if not processed_battery.empty
        else None
    )
    low_and_high = [
        record
        for record in discharge_records
        if record["capacity_review_flags"]
    ]
    excluded = [
        record
        for record in discharge_records
        if record["capacity_ah"] is None or record["capacity_ah"] <= 0
    ]
    protocol_group = next(
        group
        for group in PROTOCOL_GROUPS
        if path.stem in group["batteries"]
    )
    cutoff = protocol_group["cutoff_voltage_v"][path.stem]
    protocol_load = protocol_group["load_profile"]
    if isinstance(protocol_load, dict):
        protocol_load = protocol_load[path.stem]
    evidence_status = (
        "candidate_for_condition_specific_review_not_validated"
        if path.stem == "B0005"
        and cutoff == 2.7
        and protocol_load == "2 A constant-current discharge"
        and not low_and_high
        and not excluded
        else "unresolved_no_defensible_cell_specific_reference"
    )

    return {
        "battery_id": path.stem,
        "source_file": path.name,
        "total_cycle_operations": len(cycles),
        "operation_counts": operation_counts,
        "discharge_operation_count": len(discharge_records),
        "all_discharge_cycle_metadata": [
            {
                key: record[key]
                for key in (
                    "source_file",
                    "battery_id",
                    "source_cycle_index",
                    "start_time_components_matlab_date_vector",
                    "ambient_temperature_c",
                    "capacity_ah",
                )
            }
            for record in discharge_records
        ],
        "positive_capacity_count": len(positive),
        "nonpositive_or_missing_capacity_records": excluded,
        "capacity_summary_ah": {
            "minimum": float(np.min(values)) if values.size else None,
            "q10": float(np.quantile(values, 0.1)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "maximum": float(np.max(values)) if values.size else None,
            "below_0_5_ah_count_review_only": int((values < 0.5).sum()),
            "above_2_0_ah_count_review_only": int((values > 2.0).sum()),
        },
        "existing_first_10_positive_median_ah_not_approved": current_reference,
        "processed_csv_reference_capacity_ah_not_approved": recorded_reference,
        "current_label_range_percent_not_approved": (
            {
                "minimum": float(processed_battery["soh_percent"].min()),
                "maximum": float(processed_battery["soh_percent"].max()),
            }
            if not processed_battery.empty
            else None
        ),
        "all_capacity_review_records": low_and_high,
        "documented_protocol": {
            "readme": protocol_group["readme"],
            "ambient_temperature_c": protocol_group[
                "ambient_temperature_c"
            ],
            "discharge_load_profile": protocol_load,
            "cutoff_voltage_v": cutoff,
            "readme_eol_capacity_ah": protocol_group[
                "readme_eol_capacity_ah"
            ],
            "readme_eol_fade_percent": protocol_group[
                "readme_eol_fade_percent"
            ],
            "readme_rated_capacity_ah": protocol_group[
                "readme_rated_capacity_ah"
            ],
            "readme_unresolved_capacity_note": protocol_group.get(
                "readme_unresolved_capacity_note"
            ),
        },
        "cycle_reference_assessment": {
            "existing_soh_labels_defensible": False,
            "status": evidence_status,
            "reason": (
                "The existing target uses the first ten positive discharge "
                "capacities, which are not designated as reference tests in "
                "the source documentation. B0005 is only a candidate for "
                "condition-specific review because its documented 24 C, 2 A, "
                "2.7 V protocol matches the common Capacity field description "
                "and no capacity review flags were observed; a source-verified "
                "cell-specific baseline and prediction-time contract are still "
                "missing."
                if path.stem == "B0005"
                else
                "The current denominator is not a documented reference test. "
                "The cell's documented protocol, capacity anomalies, or both "
                "prevent a defensible cell-specific SOH reference from being "
                "established by the available files."
            ),
            "rated_capacity_2_0_ah_is_dataset_level_evidence_only": True,
            "no_soh_labels_created": True,
        },
    }


def build_report(
    battery_audits: list[dict[str, Any]],
    preparation_reports: list[dict[str, Any]],
    quality: dict[str, Any],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    raw_discharge_count = sum(
        battery["discharge_operation_count"] for battery in battery_audits
    )
    raw_positive_count = sum(
        battery["positive_capacity_count"] for battery in battery_audits
    )
    raw_excluded = [
        row
        for battery in battery_audits
        for row in battery["nonpositive_or_missing_capacity_records"]
    ]
    raw_review_rows = [
        row
        for battery in battery_audits
        for row in battery["all_capacity_review_records"]
    ]
    candidate_batteries = [
        battery["battery_id"]
        for battery in battery_audits
        if battery["cycle_reference_assessment"]["status"]
        == "candidate_for_condition_specific_review_not_validated"
    ]
    prep_excluded_count = sum(
        len(report.get("excluded_rows", [])) for report in preparation_reports
    )
    prep_discharge_count = sum(
        report.get("discharge_cycles_seen", 0)
        for report in preparation_reports
    )

    return {
        "source": "NASA PCoE Li-ion Battery Aging Dataset",
        "report_status": "complete_raw_audit_soh_training_blocked",
        "source_url": NASA_SOURCE_URL,
        "scope": {
            "raw_mat_files_audited": len(battery_audits),
            "battery_ids": [
                battery["battery_id"] for battery in battery_audits
            ],
            "raw_files_modified": False,
            "processed_labels_modified": False,
            "models_trained": False,
        },
        "reference_capacity_decision": {
            "current_soh_definition": (
                "100 * measured discharge Capacity / median of the first 10 "
                "valid positive discharge capacities per battery"
            ),
            "current_definition_approved": False,
            "soh_labels_defensible_now": False,
            "validated_battery_ids": [],
            "unresolved_battery_ids": [
                battery["battery_id"] for battery in battery_audits
            ],
            "candidate_for_further_condition_specific_review": (
                candidate_batteries
            ),
            "nominal_capacity_evidence": {
                "value_ah": 2.0,
                "source": NASA_SOURCE_URL,
                "meaning": (
                    "A documented rated-capacity denominator for a "
                    "nominal-capacity ratio, not a cell-specific measured "
                    "reference-performance result. Its use as SOH for each "
                    "battery requires matching the capacity test conditions "
                    "to the rated-capacity protocol."
                ),
                "not_used_to_generate_labels": True,
            },
            "other_reference_evidence": (
                "README1.txt states a 2 Ah rated capacity for B0005, B0006, "
                "B0007, and B0018 and says their EOL threshold is 1.4 Ah. "
                "Other group READMEs include 1.6 Ah/20% fade EOL statements, "
                "so local experimental endpoints are not uniform."
            ),
            "reason": (
                "No available record designates a particular measured "
                "discharge as a reference-performance test for each cell. "
                "The first-ten-positive median is an undocumented heuristic, "
                "and raw cycles include documented protocol differences and "
                "unresolved low or internally inconsistent capacity records."
            ),
        },
        "source_document_findings": {
            "official_nasa_page": NASA_SOURCE_URL,
            "official_page_states": [
                "Capacity experiments span distinct current loads, voltage cutoffs, and temperatures.",
                "Some cutoff voltages are below the OEM-recommended 2.7 V.",
                "Dataset-level rated capacity is described as 2 Ah, with general EOL at 1.4 Ah (30% fade).",
                "The MATLAB Capacity field is described as Ah for discharge to 2.7 V.",
                "The intended use emphasizes run-to-failure prognostics and notes differing DOD, rest duration, and intrinsic variability.",
            ],
            "local_readme_conflicts": [
                {
                    "readmes": ["README_33_34_36.txt", "README_38_39_40.txt"],
                    "finding": (
                        "These group READMEs state EOL at 1.6 Ah (20% fade), "
                        "not the official page's general 1.4 Ah/30% description."
                    ),
                },
                {
                    "readmes": [group["readme"] for group in PROTOCOL_GROUPS],
                    "finding": (
                        "Each describes the Capacity field as Ah to 2.7 V, "
                        "while cell-specific experimental cutoffs range from "
                        "2.0 V to 2.7 V."
                    ),
                },
                {
                    "readmes": ["README_41_42_43_44.txt"],
                    "finding": (
                        "Explicitly states that several capacities are very "
                        "low and their reasons have not been fully analyzed."
                    ),
                },
            ],
            "protocol_groups": PROTOCOL_GROUPS,
        },
        "raw_data_reconciliation": {
            "raw_discharge_operations": raw_discharge_count,
            "positive_capacity_records": raw_positive_count,
            "nonpositive_or_missing_capacity_records": len(raw_excluded),
            "capacity_review_threshold_counts": {
                "below_0_5_ah_review_only": sum(
                    1
                    for row in raw_review_rows
                    if "below_0.5_ah_review_only"
                    in row["capacity_review_flags"]
                ),
                "above_2_0_ah_review_only": sum(
                    1
                    for row in raw_review_rows
                    if "above_2.0_ah_review_only"
                    in row["capacity_review_flags"]
                ),
            },
            "review_threshold_note": (
                "The 0.5 Ah and 2.0 Ah cutoffs are inherited non-scientific "
                "triage flags, not exclusion rules. All positive values and "
                "source cycle provenance are retained in this report."
            ),
            "current_processed_sample_count": int(
                len(pd.read_csv(PROCESSED_CSV))
            ),
            "preparation_report_discharge_operations": prep_discharge_count,
            "preparation_report_excluded_records": prep_excluded_count,
            "quality_report_excluded_records": quality[
                "excluded_observations"
            ]["count"],
            "reconciles_with_existing_preprocessing": (
                raw_discharge_count == prep_discharge_count
                and raw_positive_count
                == len(pd.read_csv(PROCESSED_CSV))
                and len(raw_excluded) == prep_excluded_count
            ),
            "excluded_records": raw_excluded,
            "all_capacity_review_records": raw_review_rows,
            "source_manifest_training_status": source_manifest.get(
                "training_status"
            ),
        },
        "batteries": battery_audits,
        "training_gate": {
            "status": "blocked",
            "training_performed": False,
            "model_types_blocked": ["XGBoost", "LSTM", "GRU", "CNN-LSTM"],
            "required_evidence": [
                "A documented cell-specific reference-capacity rule or measured reference-performance cycle under a defined protocol.",
                "Clarification of whether low capacities represent partial tests, current/profile effects, data-acquisition artifacts, or valid measurements.",
                "Reconciliation of each cell's actual voltage cutoff with the common 2.7 V Capacity-field description.",
                "Resolution of conflicting group EOL capacities and confirmation whether 2 Ah applies as an individual-cell rated baseline.",
                "A prediction-time feature contract that excludes target capacity and same-discharge measurement proxies unavailable at prediction time.",
            ],
        },
        "trace_diagnostic_caveat": (
            "The included absolute-current trapezoid integral is calculated "
            "from sparsely sampled raw current/time vectors only as a "
            "consistency diagnostic. It is not a capacity reference, not an "
            "independent capacity measurement, and not a basis for correcting "
            "or excluding a capacity value."
        ),
    }


def main() -> None:
    required = [PROCESSED_CSV, PREPARATION_REPORT, QUALITY_REPORT]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required NASA audit input is missing: {path}")
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"NASA raw data folder is missing: {RAW_DIR}")
    manifest = json.loads(
        (ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Dataset manifest must contain a datasets list.")
    nasa_entry = next(
        (
            item
            for item in datasets
            if isinstance(item, dict)
            and item.get("source", "").startswith("NASA")
        ),
        None,
    )
    if nasa_entry is None:
        raise ValueError("NASA dataset entry is missing from the manifest.")

    raw_files = sorted(RAW_DIR.glob("B*.mat"))
    expected_raw_files = {
        Path(str(path)).stem
        for path in nasa_entry.get("raw_files", [])
        if isinstance(path, str) and Path(str(path)).suffix.lower() == ".mat"
    }
    actual_raw_files = {path.stem for path in raw_files}
    if actual_raw_files != expected_raw_files:
        raise ValueError(
            "NASA raw .mat file names do not match the dataset manifest."
        )
    missing_readmes = [
        group["readme"]
        for group in PROTOCOL_GROUPS
        if not (RAW_DIR / group["readme"]).is_file()
    ]
    if missing_readmes:
        raise FileNotFoundError(
            "Required NASA protocol README files are missing: "
            + ", ".join(missing_readmes)
        )
    battery_audits = [summarize_mat_file(path) for path in raw_files]
    preparation_reports = json.loads(
        PREPARATION_REPORT.read_text(encoding="utf-8")
    )
    quality = json.loads(QUALITY_REPORT.read_text(encoding="utf-8"))
    if not isinstance(preparation_reports, list):
        raise ValueError("NASA preparation report must be a JSON list.")
    report = build_report(
        battery_audits,
        preparation_reports,
        quality,
        nasa_entry,
    )
    report["raw_data_reconciliation"]["raw_mat_file_count"] = len(raw_files)
    if not report["raw_data_reconciliation"][
        "reconciles_with_existing_preprocessing"
    ]:
        raise ValueError(
            "Raw NASA measurements do not reconcile with the current "
            "processed CSV and preparation report."
        )
    if report["scope"]["raw_mat_files_audited"] != nasa_entry.get(
        "raw_file_count"
    ):
        raise ValueError(
            "Raw NASA .mat file count does not match the dataset manifest."
        )

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audited raw NASA batteries: {len(battery_audits)}")
    print(
        "Raw discharge operations: "
        f"{report['raw_data_reconciliation']['raw_discharge_operations']}"
    )
    print(
        "Capacity review rows (retained): "
        f"{len(report['raw_data_reconciliation']['all_capacity_review_records'])}"
    )
    print("Defensible current SOH labels: none; training remains blocked.")
    print(f"Evidence report: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
