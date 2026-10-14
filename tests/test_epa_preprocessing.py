import json

import pandas as pd
import pytest

from scripts import prepare_epa_vehicles
from scripts.prepare_epa_vehicles import TARGET, parse_epa_rows


def source_row(**updates):
    row = {
        "id": 1,
        "year": 2025,
        "make": "Test Motors",
        "baseModel": "E",
        "model": "E Long Range",
        "atvType": "EV",
        "range": "300",
        "cityE": "25",
        "highwayE": "30",
        "combE": "27",
        "evMotor": "250",
        "charge240": "8",
        "drive": "All-Wheel Drive",
        "VClass": "Small Sport Utility Vehicle",
    }
    row.update(updates)
    return row


def test_parser_keeps_pure_ev_and_range_provenance_without_range_leakage():
    rows, report = parse_epa_rows(pd.DataFrame([
        source_row(),
        source_row(id=2, atvType="Plug-in Hybrid"),
        source_row(id=3, range="0"),
        source_row(id=4, range="not-a-number"),
        source_row(id=5, range="inf"),
    ]))

    assert len(rows) == 1
    assert rows.iloc[0]["source_record_id"] == "1"
    assert rows.iloc[0]["vehicle_identity"] == "Test Motors|E"
    assert rows.iloc[0][TARGET] == 300
    assert not {"rangeCity", "rangeHwy"} & set(rows.columns)
    assert rows.iloc[0]["make"] == "Test Motors"
    assert rows.iloc[0]["model"] == "E Long Range"
    assert report["raw_record_count"] == 5
    assert report["valid_record_count"] == 1
    assert report["excluded_record_counts"] == {
        "not_pure_battery_electric": 1,
        "missing_or_invalid_range": 3,
    }


def test_parser_rejects_incomplete_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        parse_epa_rows(pd.DataFrame({"id": [1]}))


def test_parser_rejects_missing_model_identity():
    with pytest.raises(ValueError, match="lacks make or base-model identity"):
        parse_epa_rows(pd.DataFrame([source_row(make=None)]))


def test_parser_rejects_duplicate_valid_source_ids():
    with pytest.raises(ValueError, match="record IDs are not unique"):
        parse_epa_rows(pd.DataFrame([
            source_row(id=15),
            source_row(id=15, model="E Alternate"),
        ]))


def test_quality_report_counts_missing_values_and_flags_without_removal():
    rows, report = parse_epa_rows(pd.DataFrame([
        source_row(id=1, combE=None, range="29"),
        *[source_row(id=record_id) for record_id in range(2, 7)],
        source_row(id=7, range="520"),
    ]))

    assert len(rows) == 7
    quality = report["measurement_quality"]
    assert quality["combined_e_kwh_100_miles"]["missing_count"] == 1
    assert quality["epa_range_miles"]["observed_range"] == {
        "min": 29.0,
        "max": 520.0,
    }
    assert quality["epa_range_miles"]["iqr_outlier_count_review_only"] >= 1
    assert report["duplicate_source_record_count"] == 0
    assert report["duplicate_checks"] == {
        "exact_configuration_duplicate_rows": 5,
        "repeated_model_input_signature_rows": 6,
        "cross_identity_input_signature_groups": 0,
        "cross_identity_input_signature_rows": 0,
    }


def test_manifest_records_counts_provenance_and_processing_status(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"datasets": []}', encoding="utf-8")
    monkeypatch.setattr(prepare_epa_vehicles, "MANIFEST", manifest_path)
    _, report = parse_epa_rows(pd.DataFrame([source_row()]))

    prepare_epa_vehicles.update_manifest(report)
    dataset = json.loads(manifest_path.read_text(encoding="utf-8"))["datasets"][0]

    assert dataset["raw_record_count"] == 1
    assert dataset["valid_record_count"] == 1
    assert dataset["source_url"] == prepare_epa_vehicles.SOURCE_URL
    assert dataset["target_definition"]["units"] == "miles"
    assert dataset["preprocessing_status"] == "separate_source_specific_pipeline"
