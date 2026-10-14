from pathlib import Path

import scripts.prepare_nasa_soh as nasa
from scripts.prepare_nasa_soh import capacity_review_flags


def test_low_positive_capacity_is_flagged_not_excluded():
    assert capacity_review_flags(0.2) == ["below_0.5_ah_review_only"]


def test_capacity_above_nominal_is_flagged_not_excluded():
    assert capacity_review_flags(2.1) == ["above_2.0_ah_review_only"]


def test_capacity_inside_review_bounds_has_no_flag():
    assert capacity_review_flags(1.5) == []


def test_trace_diagnostics_preserve_minimum_and_integrate_paired_samples():
    data = {
        "Voltage_measured": [4.0, 3.0, 2.0],
        "Current_measured": [-2.0, -2.0, 0.0],
        "Time": [0.0, 1800.0, 3600.0],
    }

    diagnostics = nasa.discharge_trace_diagnostics(data)

    assert diagnostics["voltage_min_v"] == 2.0
    assert diagnostics["voltage_last_v"] == 2.0
    assert diagnostics["duration_s"] == 3600.0
    assert diagnostics["absolute_current_integral_ah_diagnostic"] == 1.5


def test_preprocessing_retains_positive_capacities_and_explains_zero(monkeypatch):
    def discharge(capacity, voltage):
        return {
            "type": "discharge",
            "ambient_temperature": 24.0,
            "data": {
                "Capacity": capacity,
                "Voltage_measured": voltage,
                "Current_measured": [-2.0, -2.0, 0.0],
                "Temperature_measured": [24.0, 25.0, 24.0],
                "Time": [0.0, 1800.0, 3600.0],
            },
        }

    raw = {
        "B0005": {
            "cycle": [
                {"type": "charge"},
                discharge(0.2, [4.0, 3.0, 2.0]),
                discharge(2.1, [4.0, 3.0, 2.0]),
                discharge(0.0, [4.0, 3.5, 3.2]),
            ]
        }
    }
    monkeypatch.setattr(nasa, "loadmat", lambda *args, **kwargs: raw)

    rows, report = nasa.convert_file(Path("B0005.mat"))

    assert [row["capacity_ah"] for row in rows] == [0.2, 2.1]
    assert rows[0]["capacity_review_flags"] == "below_0.5_ah_review_only"
    assert rows[1]["capacity_review_flags"] == "above_2.0_ah_review_only"
    assert len(report["excluded_rows"]) == 1
    excluded = report["excluded_rows"][0]
    assert excluded["source_cycle_index"] == 4
    assert excluded["reason"] == "non_positive_capacity"
    assert excluded["voltage_min_v"] == 3.2
    assert excluded["absolute_current_integral_ah_diagnostic"] == 1.5
