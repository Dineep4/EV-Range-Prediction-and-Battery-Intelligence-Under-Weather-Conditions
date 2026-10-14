import numpy as np
import pandas as pd
import pytest

from training.train_benchmarks import metrics, sequence_windows, split_by_identity
from training.train_epa_range import FEATURES, IDENTITY, assign_evaluation_groups


def test_identity_split_is_deterministic_and_disjoint():
    frame = pd.DataFrame(
        {
            "battery_id": [f"B{i}" for i in range(10) for _ in range(2)],
            "value": range(20),
        }
    )

    first = split_by_identity(frame, "battery_id", seed=21)
    second = split_by_identity(frame, "battery_id", seed=21)

    assert first.train_ids == second.train_ids
    assert first.validation_ids == second.validation_ids
    assert first.test_ids == second.test_ids
    assert not (set(first.train_ids) & set(first.validation_ids))
    assert not (set(first.train_ids) & set(first.test_ids))
    assert not (set(first.validation_ids) & set(first.test_ids))
    assert (
        set(first.train_ids) | set(first.validation_ids) | set(first.test_ids)
        == set(frame["battery_id"])
    )
    assert len(first.train) + len(first.validation) + len(first.test) == len(frame)


def test_identity_split_rejects_missing_or_too_few_identities():
    with pytest.raises(ValueError, match="missing values"):
        split_by_identity(
            pd.DataFrame({"battery_id": ["B1", None, "B3"]}),
            "battery_id",
        )
    with pytest.raises(ValueError, match="At least three"):
        split_by_identity(
            pd.DataFrame({"battery_id": ["B1", "B2"]}),
            "battery_id",
        )


def test_sequence_windows_do_not_cross_identity_or_cycle_gaps():
    frame = pd.DataFrame(
        [
            ("B1", 1, 1.0, 10.0),
            ("B1", 2, 2.0, 20.0),
            ("B1", 4, 4.0, 40.0),
            ("B1", 5, 5.0, 50.0),
            ("B2", 1, 11.0, 110.0),
            ("B2", 2, 12.0, 120.0),
        ],
        columns=["battery_id", "cycle_index", "feature", "target"],
    )

    windows, targets = sequence_windows(
        frame,
        "target",
        ["feature"],
        scaler=None,
        sequence_length=2,
    )

    np.testing.assert_array_equal(
        windows[:, :, 0],
        np.array([[1, 2], [4, 5], [11, 12]], dtype=np.float32),
    )
    np.testing.assert_array_equal(targets, np.array([20, 50, 120], dtype=np.float32))


def test_metrics_exclude_near_zero_rows_from_mape_and_report_count():
    result = metrics(
        np.array([0.0, 0.05, 1.0]),
        np.array([0.3, 0.1, 1.2]),
        "range",
        mape_min_abs_target=0.1,
    )

    assert result["mape"] == pytest.approx(20.0)
    assert result["mape_valid_rows"] == 1
    assert result["mape_excluded_near_zero_rows"] == 2
    assert result["r2"] is not None


def test_metrics_reject_mismatched_and_nonfinite_values():
    with pytest.raises(ValueError, match="equally sized"):
        metrics([1.0], [1.0, 2.0], "soh")
    with pytest.raises(ValueError, match="non-finite"):
        metrics([1.0, np.nan], [1.0, 2.0], "soh")


def test_epa_identical_input_signatures_share_a_split_group():
    frame = pd.DataFrame(
        [
            ("A|Model", "A", 2024, 25.0, 30.0, 27.0, 8.0, "AWD", "SUV"),
            ("A|Sibling", "A", 2025, 24.0, 29.0, 26.0, 7.0, "FWD", "Car"),
            ("B|Other", "B", 2024, 25.0, 30.0, 27.0, 8.0, "AWD", "SUV"),
            ("C|Different", "C", 2026, 23.0, 28.0, 25.0, 6.0, "FWD", "Car"),
        ],
        columns=[
            IDENTITY,
            "make",
            "year",
            "city_e_kwh_100_miles",
            "highway_e_kwh_100_miles",
            "combined_e_kwh_100_miles",
            "charge_240_hours",
            "drive",
            "vehicle_class",
        ],
    )

    groups = assign_evaluation_groups(frame)

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] == groups.iloc[2]
    assert groups.iloc[2] != groups.iloc[3]
    assert FEATURES == [
        "year",
        "city_e_kwh_100_miles",
        "highway_e_kwh_100_miles",
        "combined_e_kwh_100_miles",
        "charge_240_hours",
        "drive",
        "vehicle_class",
    ]


def test_epa_feature_signature_groups_never_cross_partitions():
    frame = pd.DataFrame(
        [
            ("A|Model", "A", 2024, 25.0, 30.0, 27.0, 8.0, "AWD", "SUV"),
            ("A|Sibling", "A", 2025, 24.0, 29.0, 26.0, 7.0, "FWD", "Car"),
            ("B|Other", "B", 2024, 25.0, 30.0, 27.0, 8.0, "AWD", "SUV"),
            ("C|Different", "C", 2026, 22.0, 28.0, 25.0, 9.0, "AWD", "SUV"),
            ("D|One", "D", 2027, 20.0, 26.0, 23.0, 6.0, "FWD", "Car"),
        ],
        columns=[
            IDENTITY,
            "make",
            "year",
            "city_e_kwh_100_miles",
            "highway_e_kwh_100_miles",
            "combined_e_kwh_100_miles",
            "charge_240_hours",
            "drive",
            "vehicle_class",
        ],
    )
    frame["evaluation_group"] = assign_evaluation_groups(frame)
    split = split_by_identity(frame, "evaluation_group", seed=9)

    assert not (
        set(split.train[IDENTITY])
        & set(split.validation[IDENTITY])
        | set(split.train[IDENTITY])
        & set(split.test[IDENTITY])
        | set(split.validation[IDENTITY])
        & set(split.test[IDENTITY])
    )
    make_partitions = [
        set(part["make"])
        for part in (split.train, split.validation, split.test)
    ]
    assert not (
        make_partitions[0] & make_partitions[1]
        | make_partitions[0] & make_partitions[2]
        | make_partitions[1] & make_partitions[2]
    )
    input_signatures = [
        set(map(tuple, part[FEATURES].to_numpy()))
        for part in (split.train, split.validation, split.test)
    ]
    assert not (input_signatures[0] & input_signatures[1])
    assert not (input_signatures[0] & input_signatures[2])
    assert not (input_signatures[1] & input_signatures[2])
