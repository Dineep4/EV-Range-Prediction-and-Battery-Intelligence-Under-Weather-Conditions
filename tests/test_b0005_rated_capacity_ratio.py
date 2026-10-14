import numpy as np
import pandas as pd
import pytest

from training.train_b0005_rated_capacity_ratio import (
    FEATURES,
    TARGET_COLUMN,
    capacity_ratio_percent,
    chronological_split,
    load_b0005_frame,
    make_pipeline,
    validate_feature_columns,
)
from sklearn.linear_model import LinearRegression


def test_target_is_capacity_to_rated_capacity_ratio_not_clipped():
    assert capacity_ratio_percent(1.4) == pytest.approx(70.0)
    assert capacity_ratio_percent(2.1) == pytest.approx(105.0)
    assert TARGET_COLUMN == "capacity_to_rated_capacity_percent"


@pytest.mark.parametrize(
    ("capacity", "rated"),
    [
        (None, 2.0),
        (0.0, 2.0),
        (-1.0, 2.0),
        (np.nan, 2.0),
        (np.inf, 2.0),
        (1.0, None),
        (1.0, 0.0),
        (1.0, -2.0),
        (1.0, np.nan),
        (1.0, np.inf),
        (1e308, 1e-308),
    ],
)
def test_target_rejects_invalid_capacity_inputs(capacity, rated):
    with pytest.raises(ValueError):
        capacity_ratio_percent(capacity, rated)


def test_feature_allowlist_rejects_target_and_target_trace_leakage():
    validate_feature_columns(FEATURES)
    for leaking_feature in (
        "capacity_ah",
        TARGET_COLUMN,
        "voltage_mean",
        "discharge_duration_s",
    ):
        with pytest.raises(ValueError, match="Target-leaking"):
            validate_feature_columns([*FEATURES, leaking_feature])


def test_chronological_split_is_contiguous_and_keeps_all_records():
    frame = pd.DataFrame({"source_cycle_index": range(1, 101), "value": range(100)})

    train, validation, test = chronological_split(
        frame,
        train_fraction=0.6,
        validation_fraction=0.2,
        minimum_partition_size=10,
    )

    assert len(train) == 60
    assert len(validation) == 20
    assert len(test) == 20
    assert train["source_cycle_index"].iloc[-1] < validation["source_cycle_index"].iloc[0]
    assert validation["source_cycle_index"].iloc[-1] < test["source_cycle_index"].iloc[0]
    assert pd.concat([train, validation, test])["source_cycle_index"].tolist() == list(
        range(1, 101)
    )


def test_chronological_split_rejects_shuffle_and_too_few_records():
    shuffled = pd.DataFrame({"source_cycle_index": [1, 3, 2]})
    with pytest.raises(ValueError, match="strictly ordered"):
        chronological_split(
            shuffled,
            minimum_partition_size=1,
        )

    small = pd.DataFrame({"source_cycle_index": range(1, 20)})
    with pytest.raises(ValueError, match="Insufficient records"):
        chronological_split(small, minimum_partition_size=10)


def test_loader_uses_only_preceding_operation_timestamps(monkeypatch, tmp_path):
    def date_vector(hour, minute=0):
        return [2020, 1, 1, hour, minute, 0]

    cycles = [
        {"type": "charge", "time": date_vector(0)},
        {
            "type": "discharge",
            "time": date_vector(0, 30),
            "data": {"Capacity": 1.9},
        },
        {"type": "charge", "time": date_vector(1)},
        {
            "type": "discharge",
            "time": date_vector(1, 30),
            "data": {"Capacity": 1.39},
        },
        {"type": "charge", "time": date_vector(2)},
        {
            "type": "discharge",
            "time": date_vector(2, 30),
            "data": {"Capacity": 1.2},
        },
    ]
    monkeypatch.setattr(
        "training.train_b0005_rated_capacity_ratio.loadmat",
        lambda *_args, **_kwargs: {"B0005": {"cycle": cycles}},
    )

    frame = load_b0005_frame(tmp_path / "B0005.mat")

    assert frame["discharge_ordinal"].tolist() == [1, 2, 3]
    assert frame["elapsed_hours_before_discharge"].tolist() == pytest.approx(
        [0.0, 1.0, 2.0]
    )
    assert frame["capacity_ah"].tolist() == [1.9, 1.39, 1.2]
    assert frame["below_documented_eol_1_4_ah"].tolist() == [False, True, True]
    assert "start_time" not in FEATURES


@pytest.mark.parametrize("capacity", [None, np.nan, 0.0, -0.1])
def test_loader_rejects_missing_or_nonpositive_source_capacity(
    monkeypatch,
    tmp_path,
    capacity,
):
    cycles = [
        {"type": "charge", "time": [2020, 1, 1, 0, 0, 0]},
        {
            "type": "discharge",
            "time": [2020, 1, 1, 0, 30, 0],
            "data": {"Capacity": capacity},
        },
    ]
    monkeypatch.setattr(
        "training.train_b0005_rated_capacity_ratio.loadmat",
        lambda *_args, **_kwargs: {"B0005": {"cycle": cycles}},
    )

    with pytest.raises(ValueError, match="will not silently drop it"):
        load_b0005_frame(tmp_path / "B0005.mat")


def test_b0005_loader_does_not_claim_other_battery_identity(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "training.train_b0005_rated_capacity_ratio.loadmat",
        lambda *_args, **_kwargs: {"B0041": {"cycle": []}},
    )

    with pytest.raises(ValueError, match="Expected B0005"):
        load_b0005_frame(tmp_path / "B0041.mat")


def test_imputer_is_fitted_on_fit_partition_only():
    pipeline = make_pipeline(LinearRegression())
    fit_features = pd.DataFrame(
        {
            "discharge_ordinal": [1.0, 2.0, 3.0],
            "elapsed_hours_before_discharge": [0.0, np.nan, 4.0],
        }
    )
    pipeline.fit(fit_features, [90.0, 85.0, 80.0])

    np.testing.assert_allclose(
        pipeline.named_steps["imputer"].statistics_,
        np.array([2.0, 2.0]),
    )
