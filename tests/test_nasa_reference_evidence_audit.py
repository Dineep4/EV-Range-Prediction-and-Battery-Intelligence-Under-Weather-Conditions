import pandas as pd

import scripts.audit_nasa_reference_evidence as audit


def test_raw_cycle_summary_preserves_capacity_provenance_and_flags(
    tmp_path,
    monkeypatch,
):
    processed_csv = tmp_path / "battery_measurements.csv"
    pd.DataFrame(
        [
            {
                "battery_id": "B0041",
                "reference_capacity_ah": 0.05,
                "soh_percent": 100.0,
            },
            {
                "battery_id": "B0041",
                "reference_capacity_ah": 0.05,
                "soh_percent": 2200.0,
            },
        ]
    ).to_csv(processed_csv, index=False)
    monkeypatch.setattr(audit, "PROCESSED_CSV", processed_csv)

    def fake_loadmat(file_name, **kwargs):
        assert str(file_name).endswith("B0041.mat")
        assert kwargs == {"squeeze_me": True, "struct_as_record": False}
        return {
            "B0041": {
                "cycle": [
                    {"type": "charge"},
                    {
                        "type": "discharge",
                        "time": [2010, 6, 16, 10, 39, 50.39],
                        "ambient_temperature": 4.0,
                        "data": {
                            "Capacity": 0.05,
                            "Voltage_measured": [4.0, 3.0, 2.0],
                            "Current_measured": [-1.0, -1.0, 0.0],
                            "Time": [0.0, 1800.0, 3600.0],
                        },
                    },
                    {
                        "type": "discharge",
                        "time": [2010, 6, 16, 11, 39, 50.39],
                        "ambient_temperature": 4.0,
                        "data": {
                            "Capacity": 1.1,
                            "Voltage_measured": [4.0, 3.0, 2.0],
                            "Current_measured": [-1.0, -1.0, 0.0],
                            "Time": [0.0, 1800.0, 3600.0],
                        },
                    },
                ]
            }
        }

    monkeypatch.setattr(
        audit,
        "loadmat",
        fake_loadmat,
    )

    summary = audit.summarize_mat_file(tmp_path / "B0041.mat")

    assert summary["total_cycle_operations"] == 3
    assert summary["operation_counts"] == {"charge": 1, "discharge": 2}
    assert summary["discharge_operation_count"] == 2
    assert summary["positive_capacity_count"] == 2
    assert summary["all_capacity_review_records"][0][
        "source_cycle_index"
    ] == 2
    assert summary["all_capacity_review_records"][0]["capacity_ah"] == 0.05
    assert summary["all_capacity_review_records"][0][
        "start_time_components_matlab_date_vector"
    ] == [2010.0, 6.0, 16.0, 10.0, 39.0, 50.39]
    assert summary["all_capacity_review_records"][0][
        "absolute_current_integral_ah_diagnostic"
    ] == 0.75
    assert summary["cycle_reference_assessment"][
        "existing_soh_labels_defensible"
    ] is False
    assert summary["cycle_reference_assessment"]["no_soh_labels_created"] is True


def test_report_keeps_all_batteries_unresolved_and_does_not_approve_training():
    battery = {
        "battery_id": "B0041",
        "discharge_operation_count": 2,
        "positive_capacity_count": 2,
        "nonpositive_or_missing_capacity_records": [],
        "all_capacity_review_records": [
            {
                "capacity_review_flags": ["below_0.5_ah_review_only"],
            }
        ],
        "cycle_reference_assessment": {
            "status": "unresolved_no_defensible_cell_specific_reference",
        },
    }

    report = audit.build_report(
        [battery],
        [{"discharge_cycles_seen": 2, "excluded_rows": []}],
        {"excluded_observations": {"count": 0}},
        {"training_status": "blocked"},
    )

    decision = report["reference_capacity_decision"]
    assert decision["current_definition_approved"] is False
    assert decision["soh_labels_defensible_now"] is False
    assert decision["validated_battery_ids"] == []
    assert decision["unresolved_battery_ids"] == ["B0041"]
    assert report["training_gate"]["status"] == "blocked"
    assert report["training_gate"]["training_performed"] is False
    assert report["raw_data_reconciliation"][
        "capacity_review_threshold_counts"
    ]["below_0_5_ah_review_only"] == 1
