import json

import pandas as pd
import pytest

import training.train_benchmarks as benchmarks
from scripts.nasa_soh_target_audit import (
    build_audit,
    first_n_positive_reference,
    target_readiness_blockers,
)


def test_first_n_reference_reproduces_provisional_median_without_approval():
    capacities = pd.Series([0.05, 0.06, 0.04, 0.07])

    with pytest.raises(ValueError, match="Need 10 positive capacities"):
        first_n_positive_reference(capacities)

    assert first_n_positive_reference(capacities, count=4) == 0.055


def test_target_audit_preserves_flagged_and_excluded_provenance():
    frame = pd.DataFrame(
        [
            {
                "battery_id": "B0041",
                "source_file": "B0041.mat",
                "source_cycle_index": cycle_index,
                "cycle_index": cycle_index,
                "capacity_ah": capacity,
                "reference_capacity_ah": 0.05,
                "soh_percent": 100.0 if cycle_index < 10 else 2200.0,
                "ambient_temperature_c": 4.0,
            }
            for cycle_index, capacity in enumerate([0.05] * 9 + [1.1], start=1)
        ]
    )
    original = frame.copy(deep=True)
    excluded = {
        "battery_id": "B0041",
        "source_file": "B0041.mat",
        "source_cycle_index": 3,
        "capacity_ah": 0.0,
        "reason": "non_positive_capacity",
    }
    flagged = {
        "battery_id": "B0041",
        "source_file": "B0041.mat",
        "source_cycle_index": 1,
        "capacity_ah": 0.05,
        "flags": ["below_0.5_ah_review_only"],
    }

    audit = build_audit(
        frame,
        [{"excluded_rows": [excluded], "review_rows": [flagged]}],
        {"capacity_review_flag_counts": {"below_0.5_ah_retained": 1}},
        {"training_status": "processed_not_training_ready"},
    )

    assert audit["target_status"] == "blocked_not_validated"
    assert audit["processed_data_modified"] is False
    assert audit["current_processed_rows"]["retained_positive_capacity_rows"] == 10
    assert audit["current_processed_rows"]["retained_flagged_positive_capacity_rows"] == 1
    assert audit["current_processed_rows"]["excluded_nonpositive_or_invalid_rows"] == 1
    assert audit["current_processed_rows"]["exclusions"] == [excluded]
    assert audit["current_target_distribution"]["maximum_label_provenance"] == {
        "battery_id": "B0041",
        "source_file": "B0041.mat",
        "source_cycle_index": 10,
        "capacity_ah": 1.1,
        "provisional_reference_capacity_ah": 0.05,
        "soh_percent": 2200.0,
    }
    assert frame.equals(original)
    assert any("prediction-time feature contract" in item for item in audit["training_blockers"])


def test_training_readiness_requires_manifest_and_audit_approval():
    manifest = {"training_status": "validated_ready_for_training"}
    audit = {"target_status": "blocked_not_validated", "training_blockers": ["unresolved"]}

    assert target_readiness_blockers(manifest, audit) == ["unresolved"]
    assert target_readiness_blockers(manifest, None) == [
        "NASA SOH target audit report is missing."
    ]
    assert target_readiness_blockers(
        {"training_status": "processed_not_training_ready"},
        {"target_status": "validated_ready_for_training"},
    ) == [
        "NASA manifest does not mark the SOH target "
        "'validated_ready_for_training'."
    ]


def test_training_guard_rejects_unapproved_target_audit(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    audit_path = tmp_path / "audit.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "source": "NASA PCoE",
                        "training_status": "validated_ready_for_training",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "target_status": "blocked_not_validated",
                "training_blockers": ["reference capacity unresolved"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmarks, "MANIFEST", manifest_path)
    monkeypatch.setattr(benchmarks, "NASA_TARGET_AUDIT", audit_path)
    data_path = tmp_path / "data.csv"
    data_path.touch()
    monkeypatch.setattr(benchmarks, "DATA", data_path)

    with pytest.raises(SystemExit, match="does not authorize training"):
        benchmarks.main()
