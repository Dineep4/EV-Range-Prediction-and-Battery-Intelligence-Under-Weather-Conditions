from zipfile import ZipFile

import pytest

from scripts.prepare_ved_ev import read_ved_ev_ids, summarize_trip


def point(timestamp_ms, *, speed=60.0, current=-1.0, soc=80.0):
    return {
        "timestamp_ms": timestamp_ms,
        "latitude": 38.0,
        "longitude": -77.0,
        "speed_kmh": speed,
        "outside_air_temperature_c": 20.0,
        "battery_current_a": current,
        "soc_percent": soc,
        "battery_voltage_v": 400.0,
    }


def make_inline_cell(reference, value):
    return (
        f'<c r="{reference}" t="inlineStr"><is><t>{value}</t></is></c>'
    )


def test_reads_pure_ev_ids_from_xlsx(tmp_path):
    workbook_path = tmp_path / "vehicles.xlsx"
    with ZipFile(workbook_path, "w") as workbook:
        workbook.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships"><sheets><sheet name="Vehicles" '
            'sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"><Relationship Id="rId1" '
            'Target="worksheets/sheet1.xml" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/worksheet"/></Relationships>',
        )
        workbook.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main"><sheetData>'
            "<row>"
            + make_inline_cell("A1", "VehId")
            + make_inline_cell("B1", "EngineType")
            + "</row><row>"
            + make_inline_cell("A2", "10")
            + make_inline_cell("B2", "EV")
            + "</row><row>"
            + make_inline_cell("A3", "20")
            + make_inline_cell("B3", "HEV")
            + "</row></sheetData></worksheet>",
        )

    assert read_ved_ev_ids(workbook_path) == {"10"}


def test_derives_energy_and_distance_for_complete_soc_decrease():
    summary = summarize_trip(
        "10",
        "trip-1",
        "week.csv",
        [
            point(0, soc=80.0),
            point(500, soc=79.9),
            point(1000, soc=79.8),
        ],
    )

    assert summary["speed_integrated_distance_km"] == pytest.approx(1 / 60)
    assert summary["speed_distance_coverage_fraction"] == 1.0
    assert summary["battery_signed_energy_kwh"] == pytest.approx(-400 / 3_600_000)
    assert summary["energy_consumed_kwh"] == pytest.approx(400 / 3_600_000)
    assert summary["energy_coverage_fraction"] == 1.0


def test_withholds_energy_label_when_intervals_exceed_coverage_limit():
    summary = summarize_trip(
        "10",
        "trip-gap",
        "week.csv",
        [point(0, soc=80.0), point(3000, soc=79.0)],
    )

    assert summary["energy_coverage_fraction"] == 0.0
    assert summary["energy_consumed_kwh"] is None
    assert summary["gap_interval_count"] == 1
    assert "energy_coverage_below_95_percent" in summary["quality_flags"]


def test_withholds_label_when_energy_sign_disagrees_with_soc():
    summary = summarize_trip(
        "10",
        "trip-sign",
        "week.csv",
        [
            point(0, current=1.0, soc=80.0),
            point(500, current=1.0, soc=79.8),
            point(1000, current=1.0, soc=79.6),
        ],
    )

    assert summary["energy_consumed_kwh"] is None
    assert "battery_energy_sign_disagrees_with_soc_change" in (
        summary["quality_flags"]
    )


def test_reports_duplicate_and_out_of_order_source_timestamps():
    summary = summarize_trip(
        "10",
        "trip-time",
        "week.csv",
        [
            point(1000, soc=80.0),
            point(0, soc=79.9),
            point(1000, soc=79.8),
        ],
    )

    assert summary["duplicate_timestamp_count"] == 1
    assert summary["source_timestamp_order_violation_count"] == 1
    assert "duplicate_timestamps" in summary["quality_flags"]
    assert "source_timestamps_not_strictly_increasing" in (
        summary["quality_flags"]
    )
