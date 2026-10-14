"""Prepare EPA vehicle-level electric-range estimates as a separate dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "EPA" / "vehicles.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "epa"
OUTPUT = OUTPUT_DIR / "epa_ev_range.csv"
QUALITY_REPORT = OUTPUT_DIR / "epa_ev_range_quality_report.json"
MANIFEST = ROOT / "data" / "dataset_manifest.json"
SOURCE_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv"
TARGET = "epa_range_miles"
NUMERIC_FEATURES = [
    "year",
    "cityE",
    "highwayE",
    "combE",
    "charge240",
]
CATEGORICAL_FEATURES = ["drive", "VClass"]
REQUIRED = [
    "id",
    "year",
    "make",
    "baseModel",
    "model",
    "atvType",
    "range",
    "evMotor",
    *NUMERIC_FEATURES[1:],
    *CATEGORICAL_FEATURES,
]
OUTPUT_COLUMNS = [
    "source_dataset",
    "source_record_id",
    "vehicle_identity",
    "year",
    "make",
    "base_model",
    "model",
    "drive",
    "vehicle_class",
    "city_e_kwh_100_miles",
    "highway_e_kwh_100_miles",
    "combined_e_kwh_100_miles",
    "ev_motor_description",
    "charge_240_hours",
    TARGET,
]


def measurement_quality(frame: pd.DataFrame) -> dict:
    summaries = {}
    for column in [
        "year",
        "city_e_kwh_100_miles",
        "highway_e_kwh_100_miles",
        "combined_e_kwh_100_miles",
        "charge_240_hours",
        TARGET,
    ]:
        values = pd.to_numeric(frame[column], errors="coerce")
        observed = values.dropna()
        if observed.empty:
            outlier_count = 0
            observed_range = {"min": None, "max": None}
        else:
            q1 = float(observed.quantile(0.25))
            q3 = float(observed.quantile(0.75))
            iqr = q3 - q1
            outlier_count = int(
                ((observed < q1 - 1.5 * iqr)
                 | (observed > q3 + 1.5 * iqr)).sum()
            )
            observed_range = {
                "min": float(observed.min()),
                "max": float(observed.max()),
            }
        summaries[column] = {
            "missing_count": int(values.isna().sum()),
            "observed_range": observed_range,
            "iqr_outlier_count_review_only": outlier_count,
        }
    summaries["ev_motor_description"] = {
        "missing_count": int(frame["ev_motor_description"].isna().sum()),
        "outlier_count": None,
        "note": "Free-text source descriptor; not model input.",
    }
    for column in ["drive", "vehicle_class"]:
        summaries[column] = {
            "missing_count": int(frame[column].isna().sum()),
            "unique_value_count": int(frame[column].nunique(dropna=True)),
        }
    return summaries


def parse_epa_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select pure BEVs and preserve EPA range estimates without imputation."""
    missing_columns = sorted(set(REQUIRED) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"EPA vehicle CSV is missing required columns: {missing_columns}")

    original_count = len(frame)
    electric = frame.loc[frame["atvType"].astype("string").str.strip().eq("EV")].copy()
    for column in NUMERIC_FEATURES + ["range"]:
        electric[column] = pd.to_numeric(electric[column], errors="coerce")
        electric[column] = electric[column].where(
            np.isfinite(electric[column]), np.nan
        )

    excluded = {
        "not_pure_battery_electric": original_count - len(electric),
        "missing_or_invalid_range": int(
            electric["range"].isna().sum()
            + (electric["range"].notna() & (electric["range"] <= 0)).sum()
        ),
    }
    valid = electric.loc[electric["range"].notna() & (electric["range"] > 0)].copy()
    valid["source_record_id"] = valid["id"].astype("string")
    valid["make"] = valid["make"].fillna("").astype("string").str.strip()
    valid["baseModel"] = (
        valid["baseModel"].fillna("").astype("string").str.strip()
    )
    valid["model"] = valid["model"].fillna("").astype("string").str.strip()
    if valid["make"].eq("").any() or valid["baseModel"].eq("").any():
        raise ValueError("EPA pure EV record lacks make or base-model identity.")
    valid["vehicle_identity"] = (
        valid["make"].astype(str) + "|" + valid["baseModel"].astype(str)
    )

    output = pd.DataFrame({
        "source_dataset": "EPA FuelEconomy.gov",
        "source_record_id": valid["source_record_id"],
        "vehicle_identity": valid["vehicle_identity"],
        "year": valid["year"],
        "make": valid["make"],
        "base_model": valid["baseModel"],
        "model": valid["model"],
        "drive": valid["drive"],
        "vehicle_class": valid["VClass"],
        "city_e_kwh_100_miles": valid["cityE"],
        "highway_e_kwh_100_miles": valid["highwayE"],
        "combined_e_kwh_100_miles": valid["combE"],
        "ev_motor_description": valid["evMotor"],
        "charge_240_hours": valid["charge240"],
        TARGET: valid["range"],
    })
    output = output.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)
    if output["source_record_id"].duplicated().any():
        raise ValueError("EPA vehicle record IDs are not unique.")

    model_input_columns = [
        "year",
        "city_e_kwh_100_miles",
        "highway_e_kwh_100_miles",
        "combined_e_kwh_100_miles",
        "charge_240_hours",
        "drive",
        "vehicle_class",
    ]
    repeated_input_groups = [
        group
        for _, group in output.groupby(
            model_input_columns, dropna=False, sort=False
        )
        if len(group) > 1
    ]
    cross_identity_input_groups = [
        group
        for group in repeated_input_groups
        if group["vehicle_identity"].nunique() > 1
    ]

    report = {
        "source": "EPA FuelEconomy.gov all-model-year vehicle data",
        "source_url": SOURCE_URL,
        "acquisition": "User-authorized local inspection; original CSV retained.",
        "raw_record_count": original_count,
        "pure_battery_electric_record_count": int(len(electric)),
        "valid_record_count": int(len(output)),
        "excluded_record_counts": excluded,
        "duplicate_source_record_count": int(
            output["source_record_id"].duplicated().sum()
        ),
        "duplicate_checks": {
            "exact_configuration_duplicate_rows": int(
                output.duplicated(
                    [
                        "make",
                        "base_model",
                        "model",
                        "year",
                        "city_e_kwh_100_miles",
                        "highway_e_kwh_100_miles",
                        "combined_e_kwh_100_miles",
                        "charge_240_hours",
                        "drive",
                        "vehicle_class",
                        TARGET,
                    ],
                    keep=False,
                ).sum()
            ),
            "repeated_model_input_signature_rows": int(
                sum(len(group) for group in repeated_input_groups)
            ),
            "cross_identity_input_signature_groups": len(
                cross_identity_input_groups
            ),
            "cross_identity_input_signature_rows": int(
                sum(len(group) for group in cross_identity_input_groups)
            ),
        },
        "vehicle_identity_count": int(output["vehicle_identity"].nunique()),
        "measurement_quality": measurement_quality(output),
        "target": {
            "column": TARGET,
            "units": "miles",
            "definition": (
                "EPA combined all-electric vehicle range estimate in the "
                "official vehicle table; standardized estimate, not observed "
                "real-world trip range."
            ),
            "observed_range": {
                "min": float(output[TARGET].min()),
                "max": float(output[TARGET].max()),
            },
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "units": {
                "year": "model year",
                "cityE": "kWh/100 miles",
                "highwayE": "kWh/100 miles",
                "combE": "kWh/100 miles",
                "charge240": "hours",
            },
            "preserved_but_excluded_fields": {
                "ev_motor_description": (
                    "Original EPA motor descriptor text; excluded from model "
                    "features because it is not a consistent numeric channel."
                ),
            },
        },
        "identity_split_group": "make|baseModel",
        "limitations": [
            "EPA estimate target is not real-world or per-trip range.",
            "Vehicle record IDs represent configurations/years, not observed individual vehicles.",
            "The source has no trip, location, ambient weather, or historical weather observations.",
            "Prediction features omit make/model identity and all range variants to avoid target leakage.",
            "Use policies/licensing were not specified on the cited download/documentation pages.",
        ],
    }
    return output, report


def update_manifest(report: dict) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), list):
        raise ValueError("Dataset manifest must contain a datasets array.")
    entry = {
        "source": report["source"],
        "source_url": SOURCE_URL,
        "status": "processed_standardized_EPA_vehicle_range_not_real_world",
        "local_raw_path": "data/raw/EPA/vehicles.csv",
        "processed_file": "data/processed/epa/epa_ev_range.csv",
        "quality_report": "data/processed/epa/epa_ev_range_quality_report.json",
        "license": (
            "The official download/documentation pages reviewed do not state "
            "a dataset-specific license or usage terms; verify before reuse."
        ),
        "raw_record_count": report["raw_record_count"],
        "valid_record_count": report["valid_record_count"],
        "excluded_record_counts": report["excluded_record_counts"],
        "duplicate_source_record_count": report["duplicate_source_record_count"],
        "duplicate_checks": report["duplicate_checks"],
        "vehicle_identity_count": report["vehicle_identity_count"],
        "target_definition": report["target"],
        "preprocessing_status": "separate_source_specific_pipeline",
        "training_status": "range_model_evaluation_pending",
    }
    datasets = [
        dataset for dataset in manifest["datasets"]
        if dataset.get("source") != report["source"]
    ]
    datasets.append(entry)
    manifest["datasets"] = datasets
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"EPA source CSV is missing: {SOURCE}. No EPA data was processed."
        )
    source_frame = pd.read_csv(SOURCE, low_memory=False)
    processed, report = parse_epa_rows(source_frame)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUTPUT, index=False)
    QUALITY_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    update_manifest(report)
    print(f"EPA source rows: {report['raw_record_count']}")
    print(f"Valid pure-EV range records: {report['valid_record_count']}")
    print(f"Make/base-model identities: {report['vehicle_identity_count']}")
    print(f"Processed data: {OUTPUT}")
    print(f"Quality report: {QUALITY_REPORT}")


if __name__ == "__main__":
    main()
