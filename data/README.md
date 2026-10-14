# Dataset preparation

## Current dataset state

This workspace contains 22 local NASA PCoE `.mat` files and their source README files under `data/raw/NASA/`. Raw files are excluded by `.gitignore` and are not copied into a public repository. Preprocessing currently produces 2,040 retained discharge rows for 22 battery IDs. CALCE and Oxford files have not been acquired. The user-authorized VED archives and EV static workbook are present under `data/raw/VED/`; they were processed locally without expanding the archives or redistributing their contents.

The machine-readable source inventory is `data/dataset_manifest.json`. NASA row-level exclusions and review flags are recorded in `data/processed/nasa_preparation_report.json`; per-field missingness, units, ranges, duplicates, IQR review counts, and SOH target quality are in `data/processed/nasa_quality_report.json`; the CSV is `data/processed/battery_measurements.csv`.

## Candidate sources

- NASA PCoE Li-ion Battery Aging datasets: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
- CALCE battery data: https://calce.umd.edu/data
- Oxford Battery Degradation Dataset 1: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac
- Vehicle Energy Dataset (VED): https://github.com/gsoh/VED/tree/6baa4963782d515a67d32a5490bd5d11f5d9bf0d

### CALCE CS2 battery cycling data

The official [CALCE data page](https://calce.umd.edu/data) describes CS2 cells rated at 1.1 Ah, CC/CV charging to 4.2 V, and a usual 2.7 V discharge cutoff. It describes constant-current cycling, varied-current profiles, changed cutoffs, and partial-cycle protocols. Most CS2 data are collections of Arbin Excel files; CS2_8 and CS2_21 are CADEX TXT. CALCE calls the data open access and requests citation for publications, but the official page does not state a dataset-specific license. File sizes, exact workbook schemas, and which channels are suitable capacity labels have not been verified. Do not infer full-cycle capacity labels from partial-cycle types.

### Oxford Battery Degradation Dataset 1

The official [Oxford record](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) identifies long-term cycling of eight Kokam SLPB533459H4 740 mAh pouch cells, published in 2017 (DOI [10.5287/bodleian:KO2kdmYGg](https://doi.org/10.5287/bodleian:KO2kdmYGg)). The record directs users to a `readme.txt`, but its public metadata does not expose the payload format or an explicit reuse license. The files and full README have not been obtained or reviewed; no parser or license assumption is appropriate yet.

### Vehicle Energy Dataset (VED)

The official [VED repository at the recorded commit](https://github.com/gsoh/VED/tree/6baa4963782d515a67d32a5490bd5d11f5d9bf0d) documents trip/vehicle IDs, timestamps, GPS, speed, outside-air temperature, and high-voltage battery current, SOC, and voltage. The two dynamic 7z archives are **82,723,769 and 93,662,910 bytes** (about 168 MiB combined), expanding to about 3.20 GB; processing streams member CSVs without expanding the archives. The user authorized local inspection only. The repository README identifies three pure EVs, all 2013 Nissan Leafs with advertised 24 kWh packs, and states Apache License 2.0 with a repository `LICENSE`; the data files do not carry a separately described license, so verify the license scope before redistribution or commercial use.

`scripts/prepare_ved_ev.py` processed 476,308 pure-EV telemetry rows into 504 trip summaries. It derived speed-integrated distance and battery energy (trapezoidal integration of current × voltage); only 48 trips have an energy label after requiring at least 95% interval coverage and agreement between the energy sign and a decrease in SOC. Those 48 labels are distributed as 11, 36, and 1 across the three vehicles, so there is not enough balanced identity-held-out evidence for a reliable generalized model. The quality report exposes the 456 unlabeled trips, data ranges, IQR review-only counts, long/duplicate/non-monotonic timestamps, and missingness; no observations are removed from that report by outlier thresholds. Trip distance and energy are derived, not supplied ground-truth labels. VED contains no direct range label, covers only three same-model EV identities, and documents outside-air temperature but not precipitation, wind, or humidity. It does not support range prediction or broad/weather-aware claims. Do not redistribute the data without resolving license scope.

To regenerate the NASA and VED datasets and run their validators/tests from the project root:

```powershell
py scripts\prepare_nasa_soh.py
py scripts\validate_nasa_soh.py
py scripts\prepare_ved_ev.py
py -m pytest tests\test_nasa_preprocessing.py tests\test_ved_preprocessing.py -q
```

Run API tests from `backend\` with `py -m pytest tests\test_api.py -q`. Run the frontend build from `frontend\` with `npm run build`. Implement parsers only after authorized files and their actual schemas/terms have been inspected.

## Harmonized file schema

Save the source-specific, validated measurements at `data/processed/battery_measurements.csv`, one chronological summary row per retained battery discharge cycle.

| Column | Meaning | Units / rules |
|---|---|---|
| `battery_id` | Stable source battery/cell identity | String; never discard this |
| `source_dataset` | Original source dataset name | Preserve provenance |
| `source_file` | Original raw file name | Preserve provenance |
| `cycle_index` | Discharge-cycle index | Count; gaps remain when a discharge row is excluded |
| `source_cycle_index` | Original cycle/operation index | Preserve source order and intervening operations |
| `capacity_ah` | Measured discharge/reference capacity | Ah; record protocol |
| `capacity_review_flags` | Non-destructive quality flags | Empty or review reasons; not an exclusion rule |
| `voltage_mean` | Mean voltage over a documented window | V |
| `current_mean` | Mean current over a documented window | A; document sign convention |
| `temperature_mean` | Mean measured cell temperature | °C; do not silently substitute ambient weather |
| `ambient_temperature_c` | Recorded test ambient temperature | °C |
| `soh_percent` | Calculated SOH target | State the reference capacity and formula |
| `soh_definition` | Target formula and reference description | Required and consistent before training |
| `range_km` | Optional vehicle-level range target | Only when supported by actual vehicle data |

Cell-aging datasets do not, by themselves, supply real-world EV range labels. Keep observed measurements, calculated targets, and estimated values distinct.

## Critical validation

- Use a documented reference capacity for each cell/battery and state the SOH formula. The current NASA median-of-first-10 reference remains explicitly provisional and is not suitable for benchmark training.
- Keep whole battery identities out of the training set when testing generalization. The included training script enforces a group holdout when there are at least three unique identities.
- If feature measurements do not exist in a dataset, do not fill them with made-up values. Adapt the source-specific training pipeline and report feature differences transparently.
- For cross-vehicle generalization, include a vehicle identity/type and hold out complete identities; battery-cell IDs are not vehicle identities.
- Range performance requires a separate suitable EV driving/energy dataset and trustworthy vehicle-level labels. Never present the physics-based dashboard range as ML ground truth.
- Record source URLs, usage terms, units, exclusions, and preprocessing decisions in the manifest and quality report.

## NASA quality findings

The extractor sees 2,043 discharge operations. It retains all 2,040 finite positive `Capacity` values; three zero values are excluded as unusable capacity labels, not erased from the audit. The report records each zero row's source operation and raw voltage/current/time diagnostics. All 210 positive rows flagged by the purely triage thresholds are retained: 202 are below 0.5 Ah (B0033: 9; B0039: 12; B0040: 1; B0041: 42; B0042-B0044: 46 each), and 8 are above 2.0 Ah (B0006: 7; B0036: 1). These thresholds are not scientific exclusion rules.

The raw evidence does **not** support one cause for all flags:

- The source README files document materially different test conditions: B0005/6/7/18 use room-temperature 2 A CC discharges with cell-specific cutoffs; B0025-28 use 24 °C, 0.05 Hz 4 A square-wave loading; B0029-32 use 43 °C and 4 A; B0033/34/36 use 24 °C with 4 A or 2 A; B0038-40 use 24/44 °C and 1/2/4 A; B0041-44 use 4 °C and 1/4 A. The cutoff voltages vary from 2.0 to 2.7 V within several groups. This protocol diversity makes unqualified cross-cell capacity and SOH comparisons unsafe.
- The local NASA README for B0041-B0044 explicitly warns that several very-low-capacity runs are unexplained. B0041's first ten capacities are about 0.044-0.056 Ah, but later capacities are about 0.84-1.22 Ah. Its first-ten median (about 0.0558 Ah) therefore creates a >2,100% ratio. This is not a credible monotonic degradation trajectory, but the available source does not establish whether the early field values are partial tests, protocol artifacts, or measurement/export errors.
- Many B0042-B0044 low readings have recorded minimum voltages near their respective documented cutoffs. For B0042 specifically, 40 of 46 sub-0.5 Ah readings have a coarse integral of the sparse `Current_measured`/`Time` samples over five times the recorded `Capacity`; B0043/B0044 do not show that same discrepancy. This is evidence of unresolved channel/protocol semantics for B0042, not proof that either signal or field is correct; the current trace is not dense enough to substitute for a calibrated capacity measurement.
- The three zero-capacity operations are source operation 15 in B0042, B0043, and B0044. Their traces each contain 274 voltage samples over about 2,647 seconds, with minimum voltages about 3.29-3.38 V—above the corresponding 2.2/2.5/2.7 V cutoffs. A sparse integral of the current trace is about 1.45 Ah, despite `Capacity` being zero. These are incomplete or internally inconsistent observations, but the evidence cannot distinguish interruption from a bad capacity field. They remain explicitly excluded from capacity-target rows with telemetry diagnostics retained in the report.
- The eight >2 Ah readings are not equivalent anomalies: seven B0006 readings are about 2.0005-2.0353 Ah around the source's 2 Ah rated capacity; one B0036 reading is 2.444 Ah. They are all retained and flagged; neither a nominal rating nor an IQR threshold proves a measurement invalid.

The preparation report now includes raw trace minima/end voltages, sample counts, durations, and an absolute-current time integral for flagged and excluded observations. The integral is expressly diagnostic only (sparse samples, not a validated capacity calculation). No positive measurement is silently removed, and no SOH is clipped.

### SOH target decision

The present `100 × capacity / median(first 10 positive capacities)` remains a diagnostic-only provisional ratio, not a justified SOH target. It is especially invalid for B0041 because the reference is taken from the unexplained low group. The official NASA description gives an EOL criterion relative to a 2 Ah rating but also states that cells used differing voltage cutoffs, depths of discharge, rest periods, and load conditions; cross-cell measured capacities are not standardized intrinsic SOH measurements.

A scientifically defensible SOH target for future training is **reference-performance-test capacity retention**: for each cell, compare measured capacity from periodic, confirmed-complete reference-performance tests against that same cell's beginning-of-life reference test, holding temperature, charge completion, discharge current/profile, and cutoff voltage constant. Declare the resulting percentage as protocol-specific capacity retention (not universal SOH), and only train on identified comparable tests with a target definition fixed before splitting. The present files do not reliably identify a common standardized reference test or explain the low-capacity runs, so they do not currently satisfy that rule. NASA measured capacity itself could support an end-of-discharge capacity task after data validation, but it is not a defensible SOH label as currently prepared. **Training remains blocked.**

Run preprocessing and validation from the project root:

```powershell
py scripts\prepare_nasa_soh.py
py scripts\validate_nasa_soh.py
```

`training/train_benchmarks.py` intentionally refuses to emit metrics while the SOH definition is provisional. Resolve the reference-capacity and protocol-comparability questions first.
