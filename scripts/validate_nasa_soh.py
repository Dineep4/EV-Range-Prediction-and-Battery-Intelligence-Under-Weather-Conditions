"""Validate NASA capacity data integrity and report SOH target readiness.

Run from project root:
    python scripts/validate_nasa_soh.py
"""
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_nasa_soh import convert_file

CSV_PATH = ROOT / "data" / "processed" / "battery_measurements.csv"
REPORT_PATH = ROOT / "data" / "processed" / "nasa_preparation_report.json"
QUALITY_REPORT_PATH = ROOT / "data" / "processed" / "nasa_quality_report.json"
TARGET_AUDIT_PATH = ROOT / "data" / "processed" / "nasa_soh_target_audit.json"
MANIFEST_PATH = ROOT / "data" / "dataset_manifest.json"
REFERENCE_AUDIT_PATH = (
    ROOT / "data" / "processed" / "nasa_soh_reference_evidence_audit.json"
)
B0005_TARGET_REVIEW_PATH = (
    ROOT / "data" / "processed" / "nasa_b0005_final_target_evidence_review.json"
)
B0005_EXPERIMENT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "experiments"
    / "nasa_b0005_rated_capacity_ratio"
    / "experiment_report.json"
)
B0005_OBSERVATIONS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "experiments"
    / "nasa_b0005_rated_capacity_ratio"
    / "observations.csv"
)
VALIDATION_OUTPUT_PATH = (
    ROOT / "data" / "processed" / "nasa_battery_health_validation.json"
)

REQUIRED = [
    "battery_id", "source_dataset", "source_file", "cycle_index", "source_cycle_index", "capacity_ah",
    "reference_capacity_ah", "soh_percent", "ambient_temperature_c",
    "voltage_mean", "voltage_min", "voltage_max", "current_mean",
    "current_abs_mean", "temperature_mean", "temperature_max",
    "discharge_duration_s", "capacity_review_flags", "soh_definition",
]
FEATURES = [
    "ambient_temperature_c", "voltage_mean", "voltage_min", "voltage_max",
    "current_mean", "current_abs_mean", "temperature_mean", "temperature_max",
    "discharge_duration_s",
]

def main():
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}\nRun python scripts\\prepare_nasa_soh.py first.")
    df = pd.read_csv(CSV_PATH)
    print("=== NASA SOH DATASET VALIDATION ===")
    print(f"File: {CSV_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Unique batteries: {df['battery_id'].nunique() if 'battery_id' in df else 'MISSING COLUMN'}")
    issues = []
    scientific_blockers = []

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        print("FAIL — missing required columns:", missing_cols)
        sys.exit(1)
    print("PASS — required columns present")

    duplicates = int(df.duplicated(["battery_id", "cycle_index"]).sum())
    print(f"Duplicate battery_id + cycle_index rows: {duplicates} — {'PASS' if duplicates == 0 else 'CHECK'}")
    if duplicates:
        issues.append(f"{duplicates} duplicate battery/cycle rows")

    cycle_gaps = int(
        df.sort_values(["battery_id", "cycle_index"])
        .groupby("battery_id")["cycle_index"]
        .diff()
        .gt(1)
        .sum()
    )
    print(
        f"Retained discharge-cycle index gaps: {cycle_gaps} — "
        f"{'REVIEW' if cycle_gaps else 'PASS'}"
    )

    bad_capacity = int((~np.isfinite(df["capacity_ah"]) | (df["capacity_ah"] <= 0)).sum())
    bad_soh = int((~np.isfinite(df["soh_percent"]) | (df["soh_percent"] <= 0)).sum())
    print(f"Invalid/non-positive capacity rows: {bad_capacity} — {'PASS' if bad_capacity == 0 else 'CHECK'}")
    print(f"Invalid/non-positive SOH rows: {bad_soh} — {'PASS' if bad_soh == 0 else 'CHECK'}")
    if bad_capacity:
        issues.append(f"{bad_capacity} invalid/non-positive capacity rows")
    if bad_soh:
        issues.append(f"{bad_soh} invalid/non-positive SOH rows")

    print("\nMissing values by feature:")
    for col in FEATURES:
        n = int(df[col].isna().sum())
        print(f"  {col}: {n} / {len(df)}")
        if n:
            issues.append(f"{n} missing values in {col}")

    print("\nCapacity and relative SOH summary:")
    print(df[["capacity_ah", "reference_capacity_ah", "soh_percent"]].describe().round(4).to_string())
    review_rows = int(df["capacity_review_flags"].fillna("").ne("").sum())
    print(f"Positive capacity rows flagged for review (retained): {review_rows}")
    print(f"SOH rows above 100%: {int((df['soh_percent'] > 100).sum())}")
    print(f"SOH rows above 120%: {int((df['soh_percent'] > 120).sum())}")
    print(f"SOH rows below 50%: {int((df['soh_percent'] < 50).sum())}")

    print("\nRows per battery:")
    print(df.groupby("battery_id").size().to_string())

    # Verify target is exactly the stated within-battery ratio.
    expected = 100.0 * df["capacity_ah"] / df["reference_capacity_ah"]
    max_error = float(np.max(np.abs(expected - df["soh_percent"]))) if len(df) else float("nan")
    print(f"\nSOH formula maximum absolute difference: {max_error:.10f} — {'PASS' if max_error < 1e-5 else 'CHECK'}")
    if not np.isfinite(max_error) or max_error >= 1e-5:
        issues.append("SOH values do not match the documented ratio")

    if REPORT_PATH.exists():
        try:
            preparation = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            errors = [r for r in preparation if r.get("status") != "ok"]
            excluded_rows = [
                row
                for entry in preparation
                for row in entry.get("excluded_rows", [])
            ]
            reported_retained = sum(
                entry.get("retained_rows", 0) for entry in preparation
            )
            reported_discharge = sum(
                entry.get("discharge_cycles_seen", 0) for entry in preparation
            )
            print(
                f"Preparation report entries: {len(preparation)}; "
                f"non-ok entries: {len(errors)}"
            )
            print(
                f"Raw discharge cycles: {reported_discharge}; retained: "
                f"{reported_retained}; excluded with reasons: "
                f"{len(excluded_rows)}"
            )
            if reported_retained != len(df):
                issues.append("report retained-row count does not match CSV")
            if reported_discharge != reported_retained + len(excluded_rows):
                issues.append("raw discharge count does not reconcile")
            if errors:
                print("Non-ok report entries (inspect these):")
                for entry in errors:
                    print(" ", entry)
                issues.append(f"{len(errors)} source files did not process successfully")
            missing_reasons = sum(not row.get("reason") for row in excluded_rows)
            if missing_reasons:
                issues.append(f"{missing_reasons} excluded rows have no reason")
            missing_trace = sum(
                not row.get("voltage_sample_count")
                or not row.get("time_sample_count")
                or row.get("voltage_min_v") is None
                for row in excluded_rows
                if row.get("reason") == "non_positive_capacity"
            )
            if missing_trace:
                issues.append(
                    f"{missing_trace} non-positive capacity exclusions lack "
                    "raw voltage/time trace diagnostics"
                )
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"Could not parse preparation report: {exc}")
    else:
        sys.exit("Preparation report not found.")

    raw_rows = []
    raw_reports = []
    raw_paths = sorted((ROOT / "data" / "raw" / "NASA").glob("*.mat"))
    if not raw_paths:
        issues.append("No raw NASA MATLAB files were found.")
    for raw_path in raw_paths:
        try:
            rows, raw_report = convert_file(raw_path)
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f"Could not re-read {raw_path.name}: {exc}")
            continue
        raw_rows.extend(rows)
        raw_reports.append(raw_report)

    raw_reconciled = False
    if raw_rows:
        raw_df = pd.DataFrame(raw_rows)
        key_columns = ["battery_id", "source_cycle_index"]
        raw_keys = pd.MultiIndex.from_frame(raw_df[key_columns])
        processed_keys = pd.MultiIndex.from_frame(df[key_columns])
        if raw_keys.has_duplicates or processed_keys.has_duplicates:
            issues.append("Raw or processed battery/source-cycle keys are duplicated.")
        elif not raw_keys.sort_values().equals(processed_keys.sort_values()):
            issues.append("Raw MATLAB source operations do not match processed CSV keys.")
        else:
            raw_by_key = raw_df.set_index(key_columns).sort_index()
            processed_by_key = df.set_index(key_columns).sort_index()
            capacities_match = np.allclose(
                raw_by_key["capacity_ah"].to_numpy(dtype=float),
                processed_by_key["capacity_ah"].to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
            )
            if not capacities_match:
                issues.append(
                    "Raw MATLAB capacities do not match processed CSV within "
                    "1e-12 Ah."
                )
            else:
                raw_reconciled = True
        raw_report_by_file = {
            entry.get("file"): entry for entry in raw_reports
        }
        prepared_by_file = {
            entry.get("file"): entry for entry in preparation
        }
        if set(raw_report_by_file) != set(prepared_by_file):
            issues.append("Raw and saved preparation reports list different source files.")
        else:
            for filename, raw_report in raw_report_by_file.items():
                prepared_report = prepared_by_file[filename]
                for key in ("discharge_cycles_seen", "retained_rows"):
                    if raw_report.get(key) != prepared_report.get(key):
                        issues.append(
                            f"{filename}: raw re-read and saved preparation "
                            f"report disagree on {key}."
                        )

    if not QUALITY_REPORT_PATH.exists():
        sys.exit("NASA quality report not found.")
    try:
        quality = json.loads(QUALITY_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Could not parse NASA quality report: {exc}")
    if not isinstance(quality, dict):
        sys.exit("NASA quality report root must be a JSON object.")
    if quality.get("sample_count") != len(df):
        issues.append("quality report sample count does not match CSV")
    if quality.get("duplicates", {}).get("count") != duplicates:
        issues.append("quality report duplicate count does not match CSV")
    if quality.get("excluded_observations", {}).get("count") != len(excluded_rows):
        issues.append("quality report exclusion count does not match preparation report")
    expected_review_flag_counts = {
        "below_0.5_ah_retained": int(df["capacity_ah"].lt(0.5).sum()),
        "above_2.0_ah_retained": int(df["capacity_ah"].gt(2.0).sum()),
    }
    if quality.get("capacity_review_flag_counts") != expected_review_flag_counts:
        issues.append("quality report review-flag counts do not match CSV")

    if not TARGET_AUDIT_PATH.exists():
        issues.append(
            "NASA SOH target audit report is missing; target readiness is "
            "not established"
        )
        target_audit = None
    else:
        try:
            target_audit = json.loads(
                TARGET_AUDIT_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"Could not parse NASA SOH target audit: {exc}")
        if not isinstance(target_audit, dict):
            sys.exit("NASA SOH target audit root must be a JSON object.")
        if target_audit.get("target_status") != "validated_ready_for_training":
            scientific_blockers.extend(
                str(reason)
                for reason in target_audit.get(
                    "training_blockers",
                    ["SOH target is not validated for training."],
                )
            )
        if target_audit.get("processed_data_modified") is not False:
            issues.append(
                "NASA audit must explicitly preserve the processed source data"
            )
        if target_audit.get("current_processed_rows", {}).get(
            "retained_positive_capacity_rows"
        ) != len(df):
            issues.append("NASA target audit retained-row count does not match CSV")

    if not MANIFEST_PATH.exists():
        sys.exit("Dataset manifest not found.")
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Could not parse dataset manifest: {exc}")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), list):
        sys.exit("Dataset manifest must contain a datasets array.")
    nasa = next(
        (
            item
            for item in manifest["datasets"]
            if isinstance(item, dict)
            and item.get("source", "").startswith("NASA")
        ),
        None,
    )
    if nasa is None:
        issues.append("dataset manifest has no NASA source entry")
    else:
        expected_manifest_counts = {
            "sample_count": len(df),
            "excluded_observation_count": len(excluded_rows),
            "review_flagged_row_count": review_rows,
        }
        for key, expected_count in expected_manifest_counts.items():
            if nasa.get(key) != expected_count:
                issues.append(
                    f"dataset manifest {key}={nasa.get(key)!r}; "
                    f"expected {expected_count}"
                )
        if nasa.get("training_status") == "validated_ready_for_training":
            if target_audit is None or target_audit.get(
                "target_status"
            ) != "validated_ready_for_training":
                issues.append(
                    "manifest cannot authorize SOH training without an "
                    "explicitly validated target audit"
                )

    print("\nIMPORTANT: capacity_ah and reference_capacity_ah are target-derived fields.")
    print("Do NOT use them as model input features when predicting soh_percent.")
    if df["soh_definition"].astype(str).str.contains("provisional", case=False).any():
        print("SOH target status: PROVISIONAL — benchmark training is blocked pending review.")
    print("Measured-reference SOH deployment status: BLOCKED.")
    for blocker in scientific_blockers:
        print(f"TRAINING BLOCKED — {blocker}")
    if issues:
        for issue in issues:
            print(f"FAIL — {issue}")
        sys.exit(1)
    print(
        "PASS — structural data and report reconciliation checks passed. "
        "This does not validate the provisional SOH target or authorize training."
    )

    reference_audit = None
    if REFERENCE_AUDIT_PATH.exists():
        try:
            reference_audit = json.loads(
                REFERENCE_AUDIT_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"Could not parse NASA reference evidence audit: {exc}")
    b0005_target_review = None
    if B0005_TARGET_REVIEW_PATH.exists():
        try:
            b0005_target_review = json.loads(
                B0005_TARGET_REVIEW_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"Could not parse B0005 target evidence review: {exc}")
    b0005_experiment = None
    if B0005_EXPERIMENT_PATH.exists():
        try:
            b0005_experiment = json.loads(
                B0005_EXPERIMENT_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"Could not parse B0005 experiment report: {exc}")
    b0005_observations = None
    if B0005_OBSERVATIONS_PATH.exists():
        try:
            b0005_observations = pd.read_csv(B0005_OBSERVATIONS_PATH)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            sys.exit(f"Could not parse B0005 experiment observations: {exc}")
    experiment_formula_verified = False
    if b0005_observations is not None:
        required_experiment_columns = {
            "battery_id",
            "source_cycle_index",
            "capacity_ah",
            "capacity_to_rated_capacity_percent",
            "below_documented_eol_1_4_ah",
        }
        if required_experiment_columns.issubset(b0005_observations.columns):
            source_b0005 = pd.DataFrame(
                [
                    row
                    for row in raw_rows
                    if row["battery_id"] == "B0005"
                ]
            )
            if not source_b0005.empty:
                observations_ordered = b0005_observations.sort_values(
                    "source_cycle_index"
                )
                source_ordered = source_b0005.sort_values("source_cycle_index")
                experiment_formula_verified = (
                    len(observations_ordered) == 168
                    and observations_ordered["battery_id"].eq("B0005").all()
                    and observations_ordered["source_cycle_index"].tolist()
                    == source_ordered["source_cycle_index"].tolist()
                    and np.allclose(
                        observations_ordered["capacity_ah"].to_numpy(dtype=float),
                        source_ordered["capacity_ah"].to_numpy(dtype=float),
                        rtol=0,
                        atol=1e-12,
                    )
                    and np.allclose(
                        observations_ordered[
                            "capacity_to_rated_capacity_percent"
                        ].to_numpy(dtype=float),
                        100.0
                        * observations_ordered["capacity_ah"].to_numpy(dtype=float)
                        / 2.0,
                        rtol=0,
                        atol=1e-10,
                    )
                    and observations_ordered[
                        "below_documented_eol_1_4_ah"
                    ].astype(bool).equals(
                        observations_ordered["capacity_ah"].lt(1.4).rename(
                            "below_documented_eol_1_4_ah"
                        )
                    )
                )
    rated_ratio_evidence = (
        b0005_target_review.get("question_findings", {})
        .get("2_ah_denominator_for_rated_capacity_ratio", {})
        if isinstance(b0005_target_review, dict)
        else {}
    )
    rated_ratio_supported = (
        rated_ratio_evidence.get("decision")
        == (
            "Supported for a B0005 capacity-to-documented-rated-capacity "
            "ratio, provided the target is named exactly as a rated-capacity "
            "ratio and the group-level rating and protocol limitations are "
            "disclosed."
        )
        and isinstance(b0005_experiment, dict)
        and b0005_experiment.get("target_name")
        == (
            "B0005 discharge capacity as percent of documented rated capacity"
        )
        and experiment_formula_verified
    )

    capacity_metadata = {}
    for _, row in df.iterrows():
        capacity_metadata[
            (str(row["battery_id"]), int(row["source_cycle_index"]))
        ] = {
            key: (
                None if pd.isna(row[key]) else float(row[key])
            )
            for key in (
                "ambient_temperature_c",
                "voltage_min",
                "voltage_max",
                "current_mean",
                "current_abs_mean",
                "temperature_mean",
                "temperature_max",
                "discharge_duration_s",
            )
        }
    flagged_observations = []
    excluded_observations = []
    protocol_by_battery = {}
    if reference_audit:
        for battery in reference_audit.get("batteries", []):
            battery_id = battery.get("battery_id")
            protocol_by_battery[battery_id] = battery.get(
                "documented_protocol", {}
            )
    for entry in raw_reports:
        for row in entry.get("review_rows", []):
            observation = dict(row)
            observation["source_field"] = "cycle.data.Capacity"
            observation["source_units"] = "Ah"
            observation["protocol"] = protocol_by_battery.get(
                row.get("battery_id"), {}
            )
            observation["processed_trace_summary"] = capacity_metadata.get(
                (
                    str(row.get("battery_id")),
                    int(row.get("source_cycle_index")),
                ),
                {},
            )
            observation["review_interpretation"] = (
                "Review-only triage flag; the positive observation is retained. "
                "The threshold alone does not establish a bad measurement."
            )
            flagged_observations.append(observation)
        for row in entry.get("excluded_rows", []):
            observation = dict(row)
            observation["source_field"] = "cycle.data.Capacity"
            observation["source_units"] = "Ah"
            observation["protocol"] = protocol_by_battery.get(
                row.get("battery_id"), {}
            )
            observation["exclusion_interpretation"] = (
                "Excluded from the positive-capacity processed table because "
                "the recorded capacity was missing, non-finite, zero, or "
                "negative; the raw operation remains recorded here."
            )
            excluded_observations.append(observation)

    target_decision = {
        "measured_reference_soh": {
            "status": "blocked_not_validated",
            "formula": (
                "100 * measured discharge capacity / justified cell-specific "
                "reference capacity"
            ),
            "approved_batteries": [],
            "reason": (
                "No reviewed NASA documentation designates a cell-specific "
                "reference-performance capacity. The first-ten positive "
                "median is an undocumented heuristic, and protocols and "
                "capacity anomalies are not resolved."
            ),
        },
        "rated_capacity_ratio": {
            "status": (
                "supported_as_distinct_b0005_metric_only"
                if rated_ratio_supported
                else "not_verified"
            ),
            "name": (
                "B0005 discharge capacity as percent of documented rated capacity"
            ),
            "formula": "100 * measured discharge Capacity (Ah) / 2.0 Ah",
            "units": "percent of rated capacity",
            "not_equivalent_to": "measured-reference SOH",
            "scope": "B0005 only; no generalization to other batteries claimed",
            "deployment_eligible": False,
            "evidence_support_verified": rated_ratio_supported,
            "observation_formula_verified": experiment_formula_verified,
            "observations_file": (
                "data/processed/experiments/nasa_b0005_rated_capacity_ratio/"
                "observations.csv"
            ),
            "experiment_methodology": (
                {
                    "scope": b0005_experiment.get("scope_and_limitations"),
                    "prediction_time_features": b0005_experiment.get(
                        "prediction_time_features"
                    ),
                    "chronological_split": b0005_experiment.get(
                        "chronological_split"
                    ),
                    "model_selection": b0005_experiment.get("model_selection"),
                    "final_test_metrics": b0005_experiment.get(
                        "final_test_metrics"
                    ),
                    "target_distribution": b0005_experiment.get(
                        "target_distribution"
                    ),
                }
                if isinstance(b0005_experiment, dict)
                else None
            ),
            "evidence_report": (
                "data/processed/nasa_b0005_final_target_evidence_review.json"
            ),
            "evidence_review_status": (
                b0005_target_review.get("report_status")
                if isinstance(b0005_target_review, dict)
                else None
            ),
            "experiment_report": (
                "data/processed/experiments/nasa_b0005_rated_capacity_ratio/"
                "experiment_report.json"
            ),
            "evaluation": (
                b0005_experiment.get("final_test_metrics", {})
                if b0005_experiment
                else None
            ),
            "limitations": [
                "The chronological test partition has negative R2.",
                "Every test observation is below the README's 1.4 Ah EOL threshold.",
                "The 44 below-threshold B0005 records are retained; their presence "
                "conflicts with the documented stopping rule and is unexplained.",
                "A single-cell chronological holdout does not validate "
                "generalization to unseen batteries.",
            ],
        },
        "overall_battery_health_model_deployment_eligible": False,
    }
    report = {
        "report_status": "structural_validation_passed_scientific_deployment_blocked",
        "source": "NASA PCoE Li-ion Battery Aging Dataset",
        "source_documentation": {
            "official_dataset_page": (
                "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets"
            ),
            "local_readmes": [
                "data/raw/NASA/README1.txt",
                "data/raw/NASA/README2.txt",
                "data/raw/NASA/README_25_26_27_28.txt",
                "data/raw/NASA/README_29_30_31_32.txt",
                "data/raw/NASA/README_33_34_36.txt",
                "data/raw/NASA/README_38_39_40.txt",
                "data/raw/NASA/README_41_42_43_44.txt",
            ],
            "previous_raw_reference_audit": (
                "data/processed/nasa_soh_reference_evidence_audit.json"
            ),
            "verified_findings": (
                reference_audit.get("source_document_findings", {})
                if isinstance(reference_audit, dict)
                else {}
            ),
            "protocols_by_battery": {
                battery.get("battery_id"): battery.get(
                    "documented_protocol", {}
                )
                for battery in (
                    reference_audit.get("batteries", [])
                    if isinstance(reference_audit, dict)
                    else []
                )
            },
        },
        "data_integrity": {
            "structural_checks_passed": True,
            "rows": int(len(df)),
            "battery_count": int(df["battery_id"].nunique()),
            "raw_mat_file_count": len(raw_paths),
            "raw_discharge_operations": int(
                    sum(item.get("discharge_cycles_seen", 0) for item in raw_reports)
                ),
                "retained_positive_capacity_rows": int(len(raw_rows)),
                "excluded_observation_count": len(excluded_observations),
                "retained_review_flagged_observation_count": len(flagged_observations),
                "duplicate_battery_cycle_rows": duplicates,
                "raw_rows_reconciled_to_processed_csv": raw_reconciled,
            "raw_exclusions_and_review_rows": {
                "all_excluded_observations": excluded_observations,
                "all_retained_flagged_observations": flagged_observations,
            },
            "original_raw_files_modified": False,
            "existing_processed_labels_modified": False,
            "existing_models_modified": False,
        },
        "provisional_label_audit": {
            "current_formula": (
                "100 * measured capacity / median of first 10 positive capacities"
            ),
            "accepted_as_measured_reference_soh": False,
            "observed_range_percent": {
                "minimum": float(df["soh_percent"].min()),
                "maximum": float(df["soh_percent"].max()),
            },
            "above_100_percent_count": int(df["soh_percent"].gt(100).sum()),
            "above_120_percent_count": int(df["soh_percent"].gt(120).sum()),
            "maximum_label_provenance": (
                target_audit.get("current_target_distribution", {}).get(
                    "maximum_label_provenance"
                )
                if target_audit
                else None
            ),
            "specific_anomaly": (
                "B0041's very low capacities are retained and flagged; the "
                "local README states the reasons have not been fully analyzed."
            ),
        },
        "target_decision": target_decision,
        "overall_decision": (
            "Option C for a deployable/general Battery Health SOH model: "
            "no measured-reference target is validated. Option B is supported "
            "only as the separately named B0005 rated-capacity research target; "
            "its existing experiment does not validate deployment or "
            "generalization."
        ),
        "existing_backend_protection": {
            "endpoint": "POST /api/predict/soh",
            "expected_status_code": 503,
            "safeguard_tested": True,
            "soh_model_deployment_eligible": False,
        },
        "excluded_observations_policy": (
            "Only non-finite or non-positive capacity entries are excluded "
            "from the current positive-capacity processed CSV; each is listed "
            "above with its source operation and reason. Review flags are not "
            "exclusions."
        ),
        "scientific_blockers": scientific_blockers,
        "structural_issues": issues,
        "report_generator": "scripts/validate_nasa_soh.py",
    }
    if VALIDATION_OUTPUT_PATH.exists():
        try:
            existing = json.loads(
                VALIDATION_OUTPUT_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(
                "Refusing to overwrite an unreadable validation report: "
                f"{exc}"
            )
        if (
            not isinstance(existing, dict)
            or existing.get("report_generator")
            != "scripts/validate_nasa_soh.py"
        ):
            sys.exit(
                "Refusing to overwrite a validation report not generated by "
                "this script."
            )
    VALIDATION_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Machine-readable validation report: {VALIDATION_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
