"""Train/evaluate a vehicle-identity-held-out model for EPA-rated range."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from training.train_benchmarks import metrics, split_by_identity


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "epa" / "epa_ev_range.csv"
QUALITY = ROOT / "data" / "processed" / "epa" / "epa_ev_range_quality_report.json"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
ARTIFACT = MODELS / "epa_range_xgboost.joblib"
RESULT = RESULTS / "epa_range_metrics.json"
TARGET = "epa_range_miles"
IDENTITY = "vehicle_identity"
NUMERIC_FEATURES = [
    "year",
    "city_e_kwh_100_miles",
    "highway_e_kwh_100_miles",
    "combined_e_kwh_100_miles",
    "charge_240_hours",
]
CATEGORICAL_FEATURES = ["drive", "vehicle_class"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SEED = 42


def assign_evaluation_groups(frame: pd.DataFrame) -> pd.Series:
    """Keep each manufacturer and identical-input aliases in one split group."""
    if "make" not in frame:
        raise ValueError("Missing manufacturer column: make")
    identities = sorted(frame[IDENTITY].astype(str).unique())
    parent = {identity: identity for identity in identities}

    def find(identity: str) -> str:
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for _, group in frame.groupby("make", dropna=False, sort=False):
        group_identities = sorted(group[IDENTITY].astype(str).unique())
        for identity in group_identities[1:]:
            union(group_identities[0], identity)

    for _, group in frame.groupby(FEATURES, dropna=False, sort=False):
        group_identities = sorted(group[IDENTITY].astype(str).unique())
        for identity in group_identities[1:]:
            union(group_identities[0], identity)

    components: dict[str, list[str]] = {}
    for identity in identities:
        components.setdefault(find(identity), []).append(identity)
    labels = {
        identity: f"EPA-GROUP-{index:03d}"
        for index, members in enumerate(
            sorted(components.values(), key=lambda values: values[0]),
            start=1,
        )
        for identity in members
    }
    return frame[IDENTITY].astype(str).map(labels).rename("evaluation_group")


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def update_manifest_training_status() -> None:
    manifest_path = ROOT / "data" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        (
            dataset for dataset in manifest.get("datasets", [])
            if dataset.get("source") == "EPA FuelEconomy.gov all-model-year vehicle data"
        ),
        None,
    )
    if entry is None:
        raise ValueError("EPA dataset manifest entry is missing.")
    entry["training_status"] = (
        "XGBoost evaluated for standardized EPA vehicle/configuration range; "
        "not real-world trip range."
    )
    entry["model_artifact"] = str(ARTIFACT.relative_to(ROOT))
    entry["evaluation_report"] = str(RESULT.relative_to(ROOT))
    entry["test_predictions"] = str(
        (RESULTS / "epa_range_test_predictions.csv").relative_to(ROOT)
    )
    entry["evaluation_split_strategy"] = (
        "manufacturer make groups, additionally joined across makes when "
        "model-input signatures are identical"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def evaluate() -> dict[str, Any]:
    if not DATA.exists() or not QUALITY.exists():
        raise FileNotFoundError(
            "EPA processed data/report are missing. Run "
            "scripts/prepare_epa_vehicles.py first."
        )
    frame = pd.read_csv(DATA)
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))
    required = {IDENTITY, TARGET, *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"EPA processed data is missing required columns: {missing}")
    frame = frame.dropna(subset=[IDENTITY, TARGET]).copy()
    if (frame[TARGET] <= 0).any() or not np.isfinite(frame[TARGET]).all():
        raise ValueError("EPA range targets must be finite positive miles.")

    frame["evaluation_group"] = assign_evaluation_groups(frame)
    split = split_by_identity(
        frame,
        "evaluation_group",
        seed=SEED,
        test_fraction=0.2,
        validation_fraction=0.2,
    )
    for name, partition in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if partition.empty:
            raise ValueError(f"EPA identity split produced an empty {name} partition.")

    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    identity_partitions = {
        name: set(partition[IDENTITY])
        for name, partition in partitions.items()
    }
    make_partitions = {
        name: set(partition["make"])
        for name, partition in partitions.items()
    }
    identity_overlap = sorted(
        (identity_partitions["train"] & identity_partitions["validation"])
        | (identity_partitions["train"] & identity_partitions["test"])
        | (identity_partitions["validation"] & identity_partitions["test"])
    )
    make_overlap = sorted(
        (make_partitions["train"] & make_partitions["validation"])
        | (make_partitions["train"] & make_partitions["test"])
        | (make_partitions["validation"] & make_partitions["test"])
    )
    group_partition = {
        group_id: name
        for name, partition in partitions.items()
        for group_id in partition["evaluation_group"].unique()
    }
    signature_overlap_count = sum(
        len({
            group_partition[group_id]
            for group_id in signature_group["evaluation_group"].unique()
        }) > 1
        for _, signature_group in frame.groupby(
            FEATURES, dropna=False, sort=False
        )
    )
    if identity_overlap or make_overlap or signature_overlap_count:
        raise ValueError(
            "EPA split integrity failed: a make, model identity, or exact "
            "feature signature crossed train, validation, and test."
        )

    preprocessor = make_preprocessor()
    train_x = preprocessor.fit_transform(split.train[FEATURES])
    validation_x = preprocessor.transform(split.validation[FEATURES])
    test_x = preprocessor.transform(split.test[FEATURES])
    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=4,
        early_stopping_rounds=30,
    )
    started = time.perf_counter()
    model.fit(
        train_x,
        split.train[TARGET],
        eval_set=[(validation_x, split.validation[TARGET])],
        verbose=False,
    )
    training_seconds = time.perf_counter() - started
    predicted = model.predict(test_x)
    result_metrics = metrics(
        split.test[TARGET].to_numpy(),
        predicted,
        "epa_vehicle_range",
    )
    result_metrics["native_units"] = "miles"
    result_metrics["native_unit_metrics"] = {
        name: result_metrics[name]
        for name in ("rmse", "mae")
    }
    for metric_name in ("rmse", "mae"):
        result_metrics[metric_name] *= 1.609344
    result_metrics.update({
        "model": "XGBoost",
        "task": "EPA vehicle-level range estimate",
        "status": "TRAINED_AND_EVALUATED",
        "target": TARGET,
        "target_units": "km",
        "target_definition": (
            quality["target"]["definition"]
            + " RMSE and MAE below are converted to kilometres."
        ),
        "range_type": "standardized EPA vehicle/configuration estimate",
        "dataset": quality["source"],
        "split_strategy": (
            "Deterministic make holdout; all identities from one manufacturer "
            "and identities sharing an exact model-input signature are grouped "
            "together. Neither a make, base identity, nor exact input signature "
            "crosses partitions."
        ),
        "seed": SEED,
        "train_records": int(len(split.train)),
        "validation_records": int(len(split.validation)),
        "test_records": int(len(split.test)),
        "train_identity_count": int(split.train[IDENTITY].nunique()),
        "validation_identity_count": int(split.validation[IDENTITY].nunique()),
        "test_identity_count": int(split.test[IDENTITY].nunique()),
        "train_make_count": int(split.train["make"].nunique()),
        "validation_make_count": int(split.validation["make"].nunique()),
        "test_make_count": int(split.test["make"].nunique()),
        "train_group_count": int(split.train["evaluation_group"].nunique()),
        "validation_group_count": int(split.validation["evaluation_group"].nunique()),
        "test_group_count": int(split.test["evaluation_group"].nunique()),
        "train_identities": sorted(split.train[IDENTITY].unique().tolist()),
        "validation_identities": sorted(split.validation[IDENTITY].unique().tolist()),
        "test_identities": sorted(split.test[IDENTITY].unique().tolist()),
        "train_evaluation_groups": split.train_ids,
        "validation_evaluation_groups": split.validation_ids,
        "test_evaluation_groups": split.test_ids,
        "identity_overlap": identity_overlap,
        "make_overlap": make_overlap,
        "exact_input_signature_overlap": [],
        "feature_schema": FEATURES,
        "feature_leakage_exclusions": [
            "make",
            "base_model",
            "model",
            "rangeCity",
            "rangeHwy",
            "rangeA",
            "evMotor",
        ],
        "best_iteration": int(model.best_iteration),
        "training_seconds": training_seconds,
        "warnings": [
            "Target is EPA's standardized model/configuration range estimate, not real-world or per-trip range.",
            "No historical weather or vehicle telemetry is present; weather-aware and remaining-range tasks are unsupported.",
            "Only XGBoost was evaluated; recurrent models are not justified for this cross-sectional vehicle-specification dataset.",
        ],
    })

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "feature_order": FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "target_units": "miles",
            "target_definition": quality["target"]["definition"],
            "dataset": quality["source"],
            "training_status": "TRAINED_AND_EVALUATED",
            "seed": SEED,
            "identity_column": IDENTITY,
            "train_identities": sorted(split.train[IDENTITY].unique().tolist()),
            "validation_identities": sorted(split.validation[IDENTITY].unique().tolist()),
            "test_identities": sorted(split.test[IDENTITY].unique().tolist()),
            "train_evaluation_groups": split.train_ids,
            "validation_evaluation_groups": split.validation_ids,
            "test_evaluation_groups": split.test_ids,
        },
        ARTIFACT,
    )
    predictions = pd.DataFrame({
        "source_record_id": split.test["source_record_id"].to_numpy(),
        "vehicle_identity": split.test[IDENTITY].to_numpy(),
        "evaluation_group": split.test["evaluation_group"].to_numpy(),
        "actual_epa_range_miles": split.test[TARGET].to_numpy(),
        "predicted_epa_range_miles": predicted,
    })
    predictions["actual_epa_range_km"] = (
        predictions["actual_epa_range_miles"] * 1.609344
    )
    predictions["predicted_epa_range_km"] = (
        predictions["predicted_epa_range_miles"] * 1.609344
    )
    predictions["absolute_error_km"] = (
        predictions["actual_epa_range_km"]
        - predictions["predicted_epa_range_km"]
    ).abs()
    predictions.to_csv(RESULTS / "epa_range_test_predictions.csv", index=False)
    report = {
        "dataset_status": "processed_standardized_EPA_vehicle_range_not_real_world",
        "dataset": quality["source"],
        "dataset_valid_record_count": quality["valid_record_count"],
        "dataset_vehicle_identity_count": quality["vehicle_identity_count"],
        "experiments": [result_metrics],
        "non_evaluated_models": [
            {
                "model": model_name,
                "task": "EPA vehicle-level range estimate",
                "status": "NOT_STARTED_NOT_SEQUENTIAL_DATA",
                "reason": (
                    "The EPA table is cross-sectional by vehicle configuration/"
                    "model year and has no per-vehicle temporal sequence inputs."
                ),
            }
            for model_name in ["LSTM", "GRU", "Hybrid CNN-LSTM"]
        ],
        "artifact": str(ARTIFACT.relative_to(ROOT)),
        "test_predictions": str(
            (RESULTS / "epa_range_test_predictions.csv").relative_to(ROOT)
        ),
        "split": {
            "method": (
                "make holdout with cross-make connected groups for identical "
                "model-input signatures"
            ),
            "seed": SEED,
            "train_identities": sorted(split.train[IDENTITY].unique().tolist()),
            "validation_identities": sorted(split.validation[IDENTITY].unique().tolist()),
            "test_identities": sorted(split.test[IDENTITY].unique().tolist()),
            "train_evaluation_groups": split.train_ids,
            "validation_evaluation_groups": split.validation_ids,
            "test_evaluation_groups": split.test_ids,
            "identity_overlap": identity_overlap,
            "make_overlap": make_overlap,
            "exact_input_signature_overlap": [],
        },
        "weather_evaluation": {
            "status": "NOT_SUPPORTED",
            "reason": "No historical weather observations are present in EPA vehicle records.",
        },
    }
    RESULT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    update_manifest_training_status()
    return report


def main() -> None:
    report = evaluate()
    experiment = report["experiments"][0]
    print(
        f"EPA XGBoost held-out RMSE={experiment['rmse']:.3f} km, "
        f"MAE={experiment['mae']:.3f} km, R2={experiment['r2']:.4f}; "
        f"native RMSE={experiment['native_unit_metrics']['rmse']:.3f} miles, "
        f"MAE={experiment['native_unit_metrics']['mae']:.3f} miles"
    )
    print(f"Model artifact: {report['artifact']}")
    print(f"Evaluation report: {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
