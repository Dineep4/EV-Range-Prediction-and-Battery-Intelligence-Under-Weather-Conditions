"""Run an isolated chronological B0005 rated-capacity-ratio experiment.

This estimates discharge capacity relative to the documented 2 Ah rating.
It is not measured-reference SOH. Outputs are written under the separate
experimental directory in data/processed/experiments/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.io import loadmat
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_nasa_soh import as_items, field, scalar


RAW_FILE = ROOT / "data" / "raw" / "NASA" / "B0005.mat"
OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "experiments"
    / "nasa_b0005_rated_capacity_ratio"
)
TARGET_NAME = "B0005 discharge capacity as percent of documented rated capacity"
TARGET_COLUMN = "capacity_to_rated_capacity_percent"
RATED_CAPACITY_AH = 2.0
FEATURES = ["discharge_ordinal", "elapsed_hours_before_discharge"]
FORBIDDEN_FEATURES = {
    "capacity_ah",
    TARGET_COLUMN,
    "soh_percent",
    "reference_capacity_ah",
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "current_mean",
    "temperature_mean",
    "discharge_duration_s",
    "elapsed_hours",
    "start_time",
}
SEED = 42


def capacity_ratio_percent(
    capacity_ah: float,
    rated_capacity_ah: float = RATED_CAPACITY_AH,
) -> float:
    """Calculate a ratio to rated capacity, without clipping the result."""
    try:
        capacity = float(capacity_ah)
        rated_capacity = float(rated_capacity_ah)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Measured and rated capacities must be finite positive numbers."
        ) from exc
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("Measured discharge capacity must be finite and positive.")
    if not np.isfinite(rated_capacity) or rated_capacity <= 0:
        raise ValueError("Rated capacity must be finite and positive.")
    ratio = 100.0 * capacity / rated_capacity
    if not np.isfinite(ratio):
        raise ValueError("Capacity-to-rated-capacity target must be finite.")
    return ratio


def validate_feature_columns(features: list[str]) -> None:
    """Fail closed if target, discharge-trace, or derived target fields enter."""
    if not features:
        raise ValueError("At least one prediction-time feature is required.")
    if len(features) != len(set(features)):
        raise ValueError("Feature names must be unique.")
    forbidden = sorted(set(features) & FORBIDDEN_FEATURES)
    if forbidden:
        raise ValueError(f"Target-leaking or post-target features: {forbidden}")
    if set(features) != set(FEATURES):
        raise ValueError(f"Only the audited prediction-time features are allowed: {FEATURES}")


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    minimum_partition_size: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make contiguous chronological partitions without shuffling observations."""
    if frame.empty:
        raise ValueError("Cannot split an empty B0005 dataset.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be strictly between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition.")
    if minimum_partition_size < 1:
        raise ValueError("minimum_partition_size must be positive.")
    if "source_cycle_index" not in frame:
        raise ValueError("Missing source_cycle_index for chronological validation.")

    indices = frame["source_cycle_index"].to_numpy(dtype=int)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("B0005 rows must be strictly ordered by source cycle index.")

    train_end = int(np.floor(len(frame) * train_fraction))
    validation_end = int(
        np.floor(len(frame) * (train_fraction + validation_fraction))
    )
    sizes = (train_end, validation_end - train_end, len(frame) - validation_end)
    if min(sizes) < minimum_partition_size:
        raise ValueError(
            "Insufficient records for chronological train/validation/test "
            f"partitions of at least {minimum_partition_size}; got {sizes}."
        )
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def _matlab_date_vector(value: Any) -> datetime:
    parts = np.asarray(value, dtype=float).reshape(-1)
    if parts.size != 6 or not np.isfinite(parts).all():
        raise ValueError("Every B0005 discharge requires a valid MATLAB date vector.")
    year, month, day, hour, minute = (int(part) for part in parts[:5])
    seconds = float(parts[5])
    return datetime(year, month, day, hour, minute) + timedelta(seconds=seconds)


def load_b0005_frame(raw_file: Path = RAW_FILE) -> pd.DataFrame:
    """Read all B0005 discharge rows and construct only pre-discharge features."""
    mat = loadmat(str(raw_file), squeeze_me=True, struct_as_record=False)
    battery_id, battery = next(
        (
            (key, mat[key])
            for key in mat
            if not key.startswith("__") and field(mat[key], "cycle") is not None
        ),
        (None, None),
    )
    if battery is None or battery_id != "B0005":
        raise ValueError(f"Expected B0005 cycle data in {raw_file}.")

    rows: list[dict[str, Any]] = []
    cycles = as_items(field(battery, "cycle"))
    first_prior_operation_time: datetime | None = None
    for zero_based_index, cycle in enumerate(cycles):
        source_cycle_index = zero_based_index + 1
        operation = field(cycle, "type", "")
        if isinstance(operation, bytes):
            operation = operation.decode("utf-8", errors="replace")
        if str(operation).strip().lower() != "discharge":
            continue
        if zero_based_index == 0:
            raise ValueError(
                "A discharge has no preceding operation timestamp for a "
                "strictly pre-discharge temporal feature."
            )
        previous_operation = cycles[zero_based_index - 1]
        previous_operation_time = _matlab_date_vector(
            field(previous_operation, "time")
        )
        start_time = _matlab_date_vector(field(cycle, "time"))
        if previous_operation_time >= start_time:
            raise ValueError(
                "The preceding operation timestamp must be earlier than "
                f"the discharge at source cycle index {source_cycle_index}."
            )
        if first_prior_operation_time is None:
            first_prior_operation_time = previous_operation_time

        capacity_ah = scalar(field(field(cycle, "data"), "Capacity"))
        if not np.isfinite(capacity_ah) or capacity_ah <= 0:
            raise ValueError(
                "B0005 contains a missing, non-finite, or non-positive "
                f"discharge capacity at source cycle index {source_cycle_index}; "
                "the experiment will not silently drop it."
            )
        rows.append({
            "battery_id": "B0005",
            "source_file": raw_file.name,
            "source_cycle_index": source_cycle_index,
            "discharge_ordinal": len(rows) + 1,
            "start_time": start_time,
            "previous_operation": str(
                field(previous_operation, "type", "")
            ).strip().lower(),
            "previous_operation_start_time": previous_operation_time,
            "elapsed_hours_before_discharge": 0.0,
            "capacity_ah": float(capacity_ah),
            TARGET_COLUMN: capacity_ratio_percent(capacity_ah),
            "below_documented_eol_1_4_ah": bool(capacity_ah < 1.4),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No B0005 discharge records were found.")
    if not frame["start_time"].is_monotonic_increasing:
        raise ValueError("B0005 discharge timestamps are not chronological.")
    frame["elapsed_hours_before_discharge"] = (
        (
            frame["previous_operation_start_time"]
            - frame["previous_operation_start_time"].iloc[0]
        ).dt.total_seconds()
        / 3600.0
    )
    validate_feature_columns(FEATURES)
    return frame


def make_pipeline(estimator: Any) -> Pipeline:
    """Keep any imputation fit inside the model's training partition."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Metrics require equally sized, non-empty arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values.")
    return {
        "rmse_percentage_points": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_percentage_points": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
    }


def _candidate_estimators() -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=4,
            random_state=SEED,
            n_jobs=1,
        ),
        "xgboost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "first_source_cycle_index": int(frame["source_cycle_index"].iloc[0]),
        "last_source_cycle_index": int(frame["source_cycle_index"].iloc[-1]),
        "first_timestamp": frame["start_time"].iloc[0].isoformat(),
        "last_timestamp": frame["start_time"].iloc[-1].isoformat(),
        "below_1_4_ah_records_retained": int(
            frame["below_documented_eol_1_4_ah"].sum()
        ),
        "target_range_percent": {
            "minimum": float(frame[TARGET_COLUMN].min()),
            "maximum": float(frame[TARGET_COLUMN].max()),
        },
    }


def run_experiment(
    raw_file: Path = RAW_FILE,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    validate_feature_columns(FEATURES)
    frame = load_b0005_frame(raw_file)
    train, validation, test = chronological_split(frame)

    train_x, train_y = train[FEATURES], train[TARGET_COLUMN].to_numpy()
    validation_x = validation[FEATURES]
    validation_y = validation[TARGET_COLUMN].to_numpy()
    candidate_scores: dict[str, dict[str, float | None]] = {}
    validation_predictions: dict[str, np.ndarray] = {}

    for name, estimator in _candidate_estimators().items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(train_x, train_y)
        prediction = pipeline.predict(validation_x)
        validation_predictions[name] = prediction
        candidate_scores[name] = regression_metrics(validation_y, prediction)

    selected_name = min(
        candidate_scores,
        key=lambda name: candidate_scores[name]["rmse_percentage_points"],
    )

    final_train = pd.concat([train, validation], ignore_index=True)
    selected_pipeline = make_pipeline(_candidate_estimators()[selected_name])
    selected_pipeline.fit(final_train[FEATURES], final_train[TARGET_COLUMN])
    final_test_predictions = selected_pipeline.predict(test[FEATURES])
    selected_test_metrics = regression_metrics(
        test[TARGET_COLUMN].to_numpy(), final_test_predictions
    )

    mean_baseline = make_pipeline(DummyRegressor(strategy="mean"))
    mean_baseline.fit(final_train[FEATURES], final_train[TARGET_COLUMN])
    baseline_test_predictions = mean_baseline.predict(test[FEATURES])
    baseline_test_metrics = regression_metrics(
        test[TARGET_COLUMN].to_numpy(), baseline_test_predictions
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "observations.csv", index=False)
    prediction_frame = frame[
        ["battery_id", "source_cycle_index", "start_time", TARGET_COLUMN]
    ].copy()
    prediction_frame["partition"] = "train"
    prediction_frame.loc[validation.index, "partition"] = "validation"
    prediction_frame.loc[test.index, "partition"] = "test"
    prediction_frame["validation_linear_regression_prediction"] = np.nan
    prediction_frame["validation_random_forest_prediction"] = np.nan
    prediction_frame["validation_xgboost_prediction"] = np.nan
    for model_name, predictions in validation_predictions.items():
        prediction_frame.loc[
            validation.index, f"validation_{model_name}_prediction"
        ] = predictions
    prediction_frame["selected_model_test_prediction"] = np.nan
    prediction_frame["mean_baseline_test_prediction"] = np.nan
    prediction_frame.loc[
        test.index, "selected_model_test_prediction"
    ] = final_test_predictions
    prediction_frame.loc[
        test.index, "mean_baseline_test_prediction"
    ] = baseline_test_predictions
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)

    test_plot = test[["source_cycle_index", TARGET_COLUMN]].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        test_plot["source_cycle_index"],
        test_plot[TARGET_COLUMN],
        marker="o",
        label="Observed",
    )
    ax.plot(
        test_plot["source_cycle_index"],
        final_test_predictions,
        marker="x",
        label=f"Predicted ({selected_name})",
    )
    ax.set_title(f"{TARGET_NAME} - chronological test holdout (B0005 only)")
    ax.set_xlabel("Original MATLAB source cycle index")
    ax.set_ylabel("Capacity / documented rated capacity (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "test_predictions.png", dpi=160)
    plt.close(fig)

    model_path = output_dir / "selected_model.joblib"
    joblib.dump(selected_pipeline, model_path)
    below_threshold_count = int(frame["below_documented_eol_1_4_ah"].sum())
    report = {
        "experiment_status": "completed_exploratory_single_battery_chronological_holdout",
        "target_name": TARGET_NAME,
        "target_definition": (
            "100 * measured discharge Capacity (Ah) / 2.0 Ah documented "
            "rated capacity"
        ),
        "target_semantics": (
            "Capacity-to-rated-capacity ratio. Not measured-reference SOH, "
            "not a cell-specific measured baseline."
        ),
        "scope_and_limitations": {
            "battery_ids": ["B0005"],
            "generalization_to_other_batteries_claimed": False,
            "records_used": int(len(frame)),
            "below_1_4_ah_records_retained": below_threshold_count,
            "below_1_4_ah_records_excluded": 0,
            "eol_discrepancy": (
                "The local README states a 1.4 Ah stopping criterion, but "
                "the raw sequence contains measurements below it; the cause "
                "remains unresolved."
            ),
            "measured_reference_soh_training_status": "blocked",
            "evaluation_interpretation": (
                "A chronological holdout from one cell measures only "
                "within-B0005 later-cycle prediction under this observed "
                "protocol; it is not an independent-cell validation."
            ),
            "test_partition_shift": (
                "The final test partition consists entirely of observations "
                "below the README's 1.4 Ah EOL threshold; the validation "
                "partition crosses that threshold. This observed tail shift "
                "limits interpretation of the holdout scores."
            ),
        },
        "prediction_time_features": {
            "columns": FEATURES,
            "definitions_and_units": {
                "discharge_ordinal": "Count of completed/target discharge position; unitless and known from operation order.",
                "elapsed_hours_before_discharge": (
                    "Elapsed hours from the first pre-discharge operation "
                    "start to the most recent completed operation start; hours."
                ),
            },
            "excluded_fields": sorted(FORBIDDEN_FEATURES),
            "contract": (
                "Features are limited to the next discharge ordinal and the "
                "timestamp of the immediately preceding completed operation. "
                "The timestamp is strictly earlier than target-discharge start. "
                "No target capacity, target discharge trace, future measurements, "
                "or target-derived fields enter X."
            ),
            "timestamp_availability_limitation": (
                "The elapsed-time feature uses the latest completed operation "
                "timestamp, not the target discharge timestamp. A deployed "
                "system must provide an equivalent prior-operation timestamp."
            ),
        },
        "chronological_split": {
            "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "minimum_partition_rows": 20,
            "partitions": {
                "train": _partition_summary(train),
                "validation": _partition_summary(validation),
                "test": _partition_summary(test),
            },
            "shuffling": False,
            "validation_use": "Candidate model selection by validation RMSE.",
            "final_fit": "Selected estimator and preprocessing refit on train + validation only; test held out.",
            "preprocessing": (
                "Median imputation is inside each sklearn Pipeline and is fit "
                "only on that fit's training partition; no feature scaler used "
                "for tree models or linear regression."
            ),
        },
        "model_selection": {
            "selection_metric": "validation RMSE in percentage points",
            "validation_metrics": candidate_scores,
            "selected_model": selected_name,
            "configurations": {
                "linear_regression": {"class": "sklearn.linear_model.LinearRegression"},
                "random_forest": {
                    "class": "sklearn.ensemble.RandomForestRegressor",
                    "n_estimators": 300,
                    "min_samples_leaf": 4,
                    "random_state": SEED,
                    "n_jobs": 1,
                },
                "xgboost": {
                    "class": "xgboost.XGBRegressor",
                    "objective": "reg:squarederror",
                    "n_estimators": 250,
                    "max_depth": 2,
                    "learning_rate": 0.03,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_lambda": 1.0,
                    "random_state": SEED,
                    "n_jobs": 1,
                },
                "mean_baseline": {
                    "class": "sklearn.dummy.DummyRegressor",
                    "strategy": "mean",
                },
            },
            "seed": SEED,
        },
        "final_test_metrics": {
            "selected_model": selected_test_metrics,
            "selected_model_name": selected_name,
            "mean_baseline": baseline_test_metrics,
            "performance_assessment": (
                "Not validated for deployment or generalization: the selected "
                "model has negative chronological test R2, and all test rows "
                "are below the documented EOL threshold. The run demonstrates "
                "pipeline execution only."
            ),
        },
        "target_distribution": {
            "minimum_percent": float(frame[TARGET_COLUMN].min()),
            "maximum_percent": float(frame[TARGET_COLUMN].max()),
            "mean_percent": float(frame[TARGET_COLUMN].mean()),
            "below_70_percent_records": int((frame[TARGET_COLUMN] < 70.0).sum()),
            "values_clipped": False,
        },
        "reproducibility": {
            "seed": SEED,
            "raw_file": raw_file.relative_to(ROOT).as_posix(),
            "raw_file_sha256": _sha256(raw_file),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": __import__("xgboost").__version__,
            "outputs": [
                "observations.csv",
                "predictions.csv",
                "test_predictions.png",
                "selected_model.joblib",
                "experiment_report.json",
            ],
        },
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, default=RAW_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run_experiment(args.raw_file, args.output_dir)
    print(json.dumps({
        "status": result["experiment_status"],
        "target_name": result["target_name"],
        "rows": result["scope_and_limitations"]["records_used"],
        "selected_model": result["model_selection"]["selected_model"],
        "test_metrics": result["final_test_metrics"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
