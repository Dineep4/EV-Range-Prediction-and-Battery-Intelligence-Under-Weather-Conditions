"""Audit whether NASA discharge capacities support training SOH labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "battery_measurements.csv"
PREPARATION_REPORT = ROOT / "data" / "processed" / "nasa_preparation_report.json"
QUALITY_REPORT = ROOT / "data" / "processed" / "nasa_quality_report.json"
MANIFEST = ROOT / "data" / "dataset_manifest.json"
OUTPUT = ROOT / "data" / "processed" / "nasa_soh_target_audit.json"

PROTOCOL_EVIDENCE = [
    {
        "readme": "README1.txt",
        "batteries": ["B0005", "B0006", "B0007", "B0018"],
        "ambient_temperature_c": 24,
        "discharge_protocol": "2 A constant-current discharge",
        "cutoff_voltage_by_battery_v": {
            "B0005": 2.7,
            "B0006": 2.5,
            "B0007": 2.2,
            "B0018": 2.5,
        },
        "rated_capacity_ah": 2.0,
        "documented_eol_capacity_ah": 1.4,
    },
    {
        "readme": "README_25_26_27_28.txt",
        "batteries": ["B0025", "B0026", "B0027", "B0028"],
        "ambient_temperature_c": 24,
        "discharge_protocol": "0.05 Hz square-wave load, 4 A amplitude, 50% duty",
        "cutoff_voltage_by_battery_v": {
            "B0025": 2.0,
            "B0026": 2.2,
            "B0027": 2.5,
            "B0028": 2.7,
        },
    },
    {
        "readme": "README_29_30_31_32.txt",
        "batteries": ["B0029", "B0030", "B0031", "B0032"],
        "ambient_temperature_c": 43,
        "discharge_protocol": "4 A discharge",
        "cutoff_voltage_by_battery_v": {
            "B0029": 2.0,
            "B0030": 2.2,
            "B0031": 2.5,
            "B0032": 2.7,
        },
    },
    {
        "readme": "README_33_34_36.txt",
        "batteries": ["B0033", "B0034", "B0036"],
        "ambient_temperature_c": 24,
        "discharge_protocol": (
            "4 A for B0033/B0034; 2 A for B0036"
        ),
        "cutoff_voltage_by_battery_v": {
            "B0033": 2.0,
            "B0034": 2.2,
            "B0036": 2.7,
        },
        "documented_eol_capacity_ah": 1.6,
    },
    {
        "readme": "README_38_39_40.txt",
        "batteries": ["B0038", "B0039", "B0040"],
        "ambient_temperature_c": [24, 44],
        "discharge_protocol": "Multiple discharge loads: 1 A, 2 A, and 4 A",
        "cutoff_voltage_by_battery_v": {
            "B0038": 2.2,
            "B0039": 2.5,
            "B0040": 2.7,
        },
        "documented_eol_capacity_ah": 1.6,
    },
    {
        "readme": "README_41_42_43_44.txt",
        "batteries": ["B0041", "B0042", "B0043", "B0044"],
        "ambient_temperature_c": 4,
        "discharge_protocol": "Multiple discharge load levels: 1 A and 4 A",
        "cutoff_voltage_by_battery_v": {
            "B0041": 2.0,
            "B0042": 2.2,
            "B0043": 2.5,
            "B0044": 2.7,
        },
        "documented_eol_capacity_ah": 1.4,
        "documented_unresolved_issue": (
            "The README says several discharge runs have very low capacity "
            "and that the reasons have not been fully analyzed."
        ),
    },
]


def first_n_positive_reference(
    capacities: pd.Series, count: int = 10
) -> float:
    """Reproduce the existing provisional reference for audit, not approval."""
    positive = pd.to_numeric(capacities, errors="coerce")
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if count < 1:
        raise ValueError("Reference sample count must be at least one.")
    if len(positive) < count:
        raise ValueError(
            f"Need {count} positive capacities; found {len(positive)}."
        )
    return float(positive.iloc[:count].median())


def summarize_batteries(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summaries = []
    ordered = frame.sort_values(["battery_id", "cycle_index"])
    for battery_id, group in ordered.groupby("battery_id", sort=True):
        reference = first_n_positive_reference(group["capacity_ah"])
        summaries.append({
            "battery_id": str(battery_id),
            "retained_positive_capacity_count": int(len(group)),
            "ambient_temperature_c_values": sorted(
                float(value)
                for value in group["ambient_temperature_c"].dropna().unique()
            ),
            "capacity_range_ah": {
                "min": float(group["capacity_ah"].min()),
                "median": float(group["capacity_ah"].median()),
                "max": float(group["capacity_ah"].max()),
            },
            "positive_capacity_below_0_5_ah_count": int(
                group["capacity_ah"].lt(0.5).sum()
            ),
            "capacity_above_2_0_ah_count": int(
                group["capacity_ah"].gt(2.0).sum()
            ),
            "current_provisional_first_10_median_ah": reference,
            "current_provisional_soh_range_percent": {
                "min": float(group["soh_percent"].min()),
                "max": float(group["soh_percent"].max()),
            },
            "reference_is_validated": False,
        })
    return summaries


def target_readiness_blockers(
    nasa_manifest: dict[str, Any] | None,
    audit: dict[str, Any] | None,
) -> list[str]:
    """Fail closed unless both records explicitly approve this target."""
    blockers = []
    if nasa_manifest is None:
        blockers.append("NASA dataset entry is missing from the manifest.")
    elif nasa_manifest.get("training_status") != "validated_ready_for_training":
        blockers.append(
            "NASA manifest does not mark the SOH target "
            "'validated_ready_for_training'."
        )

    if audit is None:
        blockers.append("NASA SOH target audit report is missing.")
    elif audit.get("target_status") != "validated_ready_for_training":
        reasons = audit.get("training_blockers")
        if isinstance(reasons, list) and reasons:
            blockers.extend(str(reason) for reason in reasons)
        else:
            blockers.append("NASA SOH target audit does not approve training.")
    return blockers


def build_audit(
    frame: pd.DataFrame,
    preparation_reports: list[dict[str, Any]],
    quality: dict[str, Any],
    nasa_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    excluded = [
        row
        for report in preparation_reports
        for row in report.get("excluded_rows", [])
    ]
    review_rows = [
        row
        for report in preparation_reports
        for row in report.get("review_rows", [])
    ]
    outlier_label = frame.loc[frame["soh_percent"].idxmax()]
    blockers = [
        "NASA documentation describes different discharge cut-off voltages, "
        "load profiles, and ambient temperatures across battery groups; "
        "capacity measurements are not protocol-equivalent.",
        "The README files describe the MATLAB Capacity field as Ah to 2.7 V, "
        "while their experiment descriptions specify cut-offs from 2.0 V to "
        "2.7 V for individual batteries.",
        "The README for B0041-B0044 explicitly says very low discharge "
        "capacities occur and their reasons have not been fully analyzed.",
        "The current first-10-positive-cycle reference is affected by "
        "unresolved low-capacity observations and has no documented healthy "
        "reference-performance measurement protocol.",
        "Current aggregate features are measured during the same discharge "
        "operation as capacity; a valid prediction-time feature contract "
        "has not been established.",
    ]
    if nasa_manifest is None or nasa_manifest.get("training_status") != (
        "validated_ready_for_training"
    ):
        blockers.append(
            "The dataset manifest does not explicitly authorize training "
            "against a validated SOH target."
        )

    return {
        "source": "NASA PCoE Li-ion Battery Aging Dataset",
        "status": "audit_complete_target_unresolved",
        "target_status": "blocked_not_validated",
        "processed_data_modified": False,
        "current_target": {
            "column": "soh_percent",
            "units": "%",
            "definition": (
                "100 * discharge Capacity / median of the first 10 valid "
                "positive discharge capacities per battery"
            ),
            "validation_status": "provisional_not_scientifically_validated",
            "important_note": (
                "The ratio is reproduced for audit only. It is not approved "
                "as an SOH label or training target."
            ),
        },
        "source_measurement": {
            "raw_mat_file_count": len(list((ROOT / "data" / "raw" / "NASA").glob("*.mat"))),
            "raw_discharge_cycle_count": int(
                sum(report.get("discharge_cycles_seen", 0)
                    for report in preparation_reports)
            ),
            "capacity_field_units": "Ah (as stated in NASA README files)",
            "documented_capacity_field_description": (
                "Battery capacity for discharge till 2.7 V"
            ),
            "protocol_evidence": PROTOCOL_EVIDENCE,
            "capacity_field_cutoff_note": (
                "The common field description says 2.7 V, but individual "
                "battery protocol descriptions specify different cut-offs; "
                "the available files do not establish a harmonized capacity "
                "measurement across those protocols."
            ),
        },
        "current_processed_rows": {
            "retained_positive_capacity_rows": int(len(frame)),
            "retained_flagged_positive_capacity_rows": len(review_rows),
            "flag_counts": quality.get("capacity_review_flag_counts", {}),
            "excluded_nonpositive_or_invalid_rows": len(excluded),
            "exclusions": excluded,
            "unresolved_positive_flagged_examples": review_rows[:20],
            "all_flagged_positive_rows_are_preserved_in_csv": True,
        },
        "current_target_distribution": {
            "min_percent": float(frame["soh_percent"].min()),
            "max_percent": float(frame["soh_percent"].max()),
            "above_100_percent_count": int(frame["soh_percent"].gt(100).sum()),
            "above_120_percent_count": int(frame["soh_percent"].gt(120).sum()),
            "below_50_percent_count": int(frame["soh_percent"].lt(50).sum()),
            "maximum_label_provenance": {
                "battery_id": str(outlier_label["battery_id"]),
                "source_file": str(outlier_label["source_file"]),
                "source_cycle_index": int(outlier_label["source_cycle_index"]),
                "capacity_ah": float(outlier_label["capacity_ah"]),
                "provisional_reference_capacity_ah": float(
                    outlier_label["reference_capacity_ah"]
                ),
                "soh_percent": float(outlier_label["soh_percent"]),
            },
        },
        "per_battery_capacity_audit": summarize_batteries(frame),
        "target_leakage_audit": {
            "capacity_ah": (
                "Direct source measurement used to calculate the target; "
                "must never be an input when predicting this SOH label."
            ),
            "reference_capacity_ah": (
                "Target-defining denominator; must never be an input."
            ),
            "soh_percent": "Target; must never be an input.",
            "current_training_features": [
                "cycle_index",
                "voltage_mean",
                "current_mean",
                "temperature_mean",
            ],
            "feature_timing_caveat": (
                "Voltage/current/temperature aggregates come from the same "
                "discharge operation as the capacity target. They are not "
                "direct target columns, but could encode the same test result; "
                "a prediction-time contract and leakage analysis are required "
                "before training."
            ),
            "training_authorized": False,
        },
        "training_blockers": blockers,
        "next_evidence_needed": [
            "Authoritative, battery-specific reference-performance capacity "
            "measurements taken under a documented common protocol.",
            "Clarification of why very low capacities occur in B0041-B0044 "
            "and whether affected runs are partial tests, protocol effects, "
            "or measurement/extraction anomalies.",
            "Resolution of the conflict between each README's 2.7 V Capacity "
            "field description and battery-specific cut-off protocols.",
            "A prediction-time feature definition that excludes capacity and "
            "other unavailable or same-measurement target proxies.",
        ],
    }


def main() -> None:
    for path in [DATA, PREPARATION_REPORT, QUALITY_REPORT, MANIFEST]:
        if not path.exists():
            raise FileNotFoundError(f"Required NASA audit input is missing: {path}")

    frame = pd.read_csv(DATA)
    preparation_reports = json.loads(
        PREPARATION_REPORT.read_text(encoding="utf-8")
    )
    quality = json.loads(QUALITY_REPORT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(preparation_reports, list):
        raise ValueError("NASA preparation report must be a JSON array.")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Dataset manifest must contain a datasets array.")
    nasa = next(
        (
            entry
            for entry in datasets
            if isinstance(entry, dict)
            and entry.get("source", "").startswith("NASA")
        ),
        None,
    )
    result = build_audit(frame, preparation_reports, quality, nasa)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"NASA target status: {result['target_status']}")
    print(
        "Rows: "
        f"{result['current_processed_rows']['retained_positive_capacity_rows']} "
        f"retained, "
        f"{result['current_processed_rows']['retained_flagged_positive_capacity_rows']} "
        "flagged and retained, "
        f"{result['current_processed_rows']['excluded_nonpositive_or_invalid_rows']} "
        "excluded."
    )
    print(f"Audit report: {OUTPUT.relative_to(ROOT)}")
    for blocker in result["training_blockers"]:
        print(f"TRAINING BLOCKED — {blocker}")


if __name__ == "__main__":
    main()
