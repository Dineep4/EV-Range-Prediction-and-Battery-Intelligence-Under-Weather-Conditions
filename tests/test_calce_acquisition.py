import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "dataset_manifest.json"
QUALITY_REPORT = (
    ROOT / "data" / "processed" / "CALCE" / "calce_pl_0_60_quality_report.json"
)


def test_calce_report_preserves_source_provenance_and_documented_units():
    report = json.loads(QUALITY_REPORT.read_text(encoding="utf-8"))

    assert report["source_url"] == "https://calce.umd.edu/battery-data"
    assert report["archive"]["http_status"] == 200
    assert report["archive"]["zip_integrity_check"] == "passed"
    assert report["documented_test_conditions"]["cell_ids"] == ["PL03", "PL10"]
    assert report["documented_measurements"]["current"] == (
        "A; positive for charge and negative for discharge"
    )
    assert report["documented_measurements"]["discharge"] == "cumulative Ah"


def test_calce_report_keeps_unread_measurements_and_labels_blocked():
    report = json.loads(QUALITY_REPORT.read_text(encoding="utf-8"))

    assert report["measurement_sample_count"] is None
    assert report["candidate_soh_definition"]["status"] == "candidate_only_not_generated"
    assert report["training_status"] == "blocked"
    assert report["cycle_level_quality_checks"]["duplicate_cycles"] == "not evaluated"
    assert report["cycle_level_quality_checks"]["capacity_value_ranges"] == "not evaluated"
    assert report["raw_files"][0]["operation_block_count"] == 24
    assert report["raw_files"][0]["known_missing_data_operation_blocks"] == 2
    assert report["raw_files"][1]["operation_block_count"] == 24
    assert report["raw_files"][1]["known_missing_data_operation_blocks"] == 2


def test_calce_manifest_status_does_not_change_nasa_training_gate_or_epa_record():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    datasets = manifest["datasets"]
    nasa = next(item for item in datasets if item["source"].startswith("NASA"))
    calce = next(
        item
        for item in datasets
        if item["source"].startswith("CALCE PL pouch-cell")
    )
    epa = next(
        item
        for item in datasets
        if item["source"].startswith("EPA FuelEconomy.gov")
    )

    assert nasa["training_status"].startswith("Blocked:")
    assert calce["training_status"].startswith("Blocked:")
    assert calce["sample_count"] is None
    assert epa["training_status"].startswith("XGBoost evaluated")
    assert epa["model_artifact"] == "models\\epa_range_xgboost.joblib"
    assert epa["evaluation_report"] == "results\\epa_range_metrics.json"
