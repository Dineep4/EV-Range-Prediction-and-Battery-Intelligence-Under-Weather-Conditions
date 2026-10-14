"""Train SOH and optional range baselines with battery-identity-aware holdout.

Input: data/processed/battery_measurements.csv with columns listed in README.
The script refuses random-row splits and refuses range evaluation without real range_km labels.
PyTorch is required for LSTM/GRU/Hybrid CNN-LSTM; XGBoost is required for XGBoost.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import json, sys, warnings
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "battery_measurements.csv"
MANIFEST = ROOT / "data" / "dataset_manifest.json"
NASA_TARGET_AUDIT = ROOT / "data" / "processed" / "nasa_soh_target_audit.json"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FEATURES = ["cycle_index", "voltage_mean", "current_mean", "temperature_mean"]
MIN_GROUPS = 3
SEED = 42


@dataclass(frozen=True)
class IdentitySplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]


def split_by_identity(
    frame: pd.DataFrame,
    identity_column: str,
    *,
    seed: int = SEED,
    test_fraction: float = 0.25,
    validation_fraction: float = 0.2,
) -> IdentitySplit:
    """Split complete identities into train, validation, and final test sets."""
    if identity_column not in frame:
        raise ValueError(f"Missing identity column: {identity_column}")
    if frame[identity_column].isna().any():
        raise ValueError(f"Identity column contains missing values: {identity_column}")
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be strictly between zero and one.")

    data = frame.copy()
    data[identity_column] = data[identity_column].astype(str)
    identities = sorted(data[identity_column].unique())
    if len(identities) < 3:
        raise ValueError(
            "At least three distinct identities are required for train, "
            "validation, and test partitions."
        )

    test_count = max(1, int(np.ceil(len(identities) * test_fraction)))
    validation_count = max(
        1,
        int(np.ceil((len(identities) - test_count) * validation_fraction)),
    )
    if test_count + validation_count >= len(identities):
        raise ValueError("Split fractions leave no identities for training.")

    shuffled = np.asarray(identities, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    test_ids = sorted(str(value) for value in shuffled[:test_count])
    validation_ids = sorted(
        str(value)
        for value in shuffled[test_count:test_count + validation_count]
    )
    train_ids = sorted(
        str(value) for value in shuffled[test_count + validation_count:]
    )

    return IdentitySplit(
        train=data[data[identity_column].isin(train_ids)].copy(),
        validation=data[data[identity_column].isin(validation_ids)].copy(),
        test=data[data[identity_column].isin(test_ids)].copy(),
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )


def sequence_windows(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    scaler: Any,
    *,
    identity_column: str = "battery_id",
    cycle_column: str = "cycle_index",
    sequence_length: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build chronological windows that never cross identities or cycle gaps."""
    required = {identity_column, cycle_column, target, *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing sequence columns: {missing}")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least one.")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    ordered = frame.sort_values([identity_column, cycle_column])
    for _, group in ordered.groupby(identity_column, sort=False):
        cycles = group[cycle_column].to_numpy(dtype=float)
        feature_values = group[features].to_numpy(dtype=float)
        if scaler is not None:
            feature_values = scaler.transform(feature_values)
        target_values = group[target].to_numpy(dtype=float)

        for end in range(sequence_length - 1, len(group)):
            start = end - sequence_length + 1
            cycle_window = cycles[start:end + 1]
            feature_window = feature_values[start:end + 1]
            label = target_values[end]
            if (
                np.all(np.diff(cycle_window) == 1)
                and np.isfinite(feature_window).all()
                and np.isfinite(label)
            ):
                windows.append(feature_window)
                targets.append(float(label))

    if not windows:
        return (
            np.empty((0, sequence_length, len(features)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.asarray(windows, dtype=np.float32), np.asarray(
        targets, dtype=np.float32
    )


def metrics(y, prediction, task, *, mape_min_abs_target: float = 1e-8):
    actual = np.asarray(y, dtype=float).reshape(-1)
    predicted = np.asarray(prediction, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Metrics require equally sized, non-empty target arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics cannot be calculated for non-finite values.")
    if mape_min_abs_target < 0:
        raise ValueError("mape_min_abs_target cannot be negative.")

    mape_mask = np.abs(actual) > mape_min_abs_target
    mape = (
        float(np.mean(np.abs((actual[mape_mask] - predicted[mape_mask])
                             / actual[mape_mask])) * 100)
        if mape_mask.any()
        else None
    )
    out = {
        "task": task,
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": mape,
        "mape_valid_rows": int(mape_mask.sum()),
        "mape_excluded_near_zero_rows": int((~mape_mask).sum()),
        "mape_min_abs_target": mape_min_abs_target,
    }
    if len(actual) >= 2 and np.var(actual) > 1e-12:
        out["r2"] = float(r2_score(actual, predicted))
    else:
        out["r2"] = None
    return out


def main():
    if not DATA.exists():
        sys.exit(f"Missing {DATA}. Prepare a harmonized CSV first; see data/README.md. No metrics were generated.")
    if not MANIFEST.exists():
        sys.exit(f"Missing {MANIFEST}. Dataset readiness must be documented before training. No metrics were generated.")
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Could not read dataset manifest: {exc}. No metrics were generated.")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), list):
        sys.exit("Dataset manifest must contain a datasets array. No metrics were generated.")
    nasa = next(
        (
            item
            for item in manifest["datasets"]
            if isinstance(item, dict)
            and item.get("source", "").startswith("NASA")
        ),
        None,
    )
    if nasa is None:
        sys.exit(
            "NASA dataset entry is missing from the manifest. SOH training "
            "fails closed; no metrics were generated."
        )
    if nasa.get("training_status") != "validated_ready_for_training":
        sys.exit(
            "NASA SOH target is not explicitly validated and training-ready "
            "in the dataset manifest. No metrics were generated."
        )
    if not NASA_TARGET_AUDIT.exists():
        sys.exit(
            "NASA SOH target audit is missing. Run "
            "scripts/nasa_soh_target_audit.py; no metrics were generated."
        )
    try:
        target_audit = json.loads(
            NASA_TARGET_AUDIT.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(
            f"Could not read NASA SOH target audit: {exc}. "
            "No metrics were generated."
        )
    if (
        not isinstance(target_audit, dict)
        or target_audit.get("target_status") != "validated_ready_for_training"
    ):
        blockers = (
            target_audit.get("training_blockers", [])
            if isinstance(target_audit, dict)
            else []
        )
        detail = "; ".join(str(reason) for reason in blockers)
        suffix = f" Blockers: {detail}" if detail else ""
        sys.exit(
            "NASA SOH target audit does not authorize training."
            f"{suffix} No metrics were generated."
        )
    df = pd.read_csv(DATA)
    required = {"battery_id", *FEATURES, "soh_percent", "soh_definition"}
    missing = sorted(required - set(df.columns))
    if missing: sys.exit(f"Missing required columns: {missing}")
    if df["soh_definition"].isna().any() or df["soh_definition"].astype(str).str.strip().eq("").any():
        sys.exit("Every SOH row must declare its target definition. No metrics were generated.")
    definitions = {
        value.strip()
        for value in df["soh_definition"].dropna().astype(str)
        if value.strip()
    }
    if len(definitions) != 1:
        sys.exit(
            "SOH rows must have one explicit, consistent target definition. "
            "No metrics were generated."
        )
    if "provisional" in next(iter(definitions)).lower():
        sys.exit(
            "The current SOH target is provisional and not ready for benchmark "
            "training. Review its reference capacity and test-protocol "
            "comparability first. No metrics were generated."
        )
    df = df.dropna(subset=list(required)).copy()
    df["battery_id"] = df["battery_id"].astype(str)
    groups = sorted(df.battery_id.unique())
    if len(groups) < MIN_GROUPS:
        sys.exit(f"Need at least {MIN_GROUPS} distinct battery_id groups for a meaningful identity holdout; found {len(groups)}. No metrics were generated.")
    split = split_by_identity(df, "battery_id", seed=SEED)
    train, validation, test = split.train, split.validation, split.test
    print(
        f"Train batteries={len(split.train_ids)}, "
        f"validation batteries={len(split.validation_ids)}, "
        f"test batteries={len(split.test_ids)}, test rows={len(test)}"
    )
    x_scaler = StandardScaler().fit(train[FEATURES])
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    experiments = []

    # XGBoost baseline
    try:
        from xgboost import XGBRegressor
        target = "soh_percent"
        tr_task = train.dropna(subset=[target])
        val_task = validation.dropna(subset=[target])
        te_task = test.dropna(subset=[target])
        if tr_task.empty or val_task.empty or te_task.empty:
            print(
                "Skipping SOH XGBoost: labeled rows are required in train, "
                "validation, and test identities."
            )
        else:
            model = XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=.04,
                subsample=.85,
                colsample_bytree=.85,
                objective="reg:squarederror",
                random_state=SEED,
                n_jobs=4,
                early_stopping_rounds=25,
            )
            model.fit(
                tr_task[FEATURES],
                tr_task[target],
                eval_set=[(val_task[FEATURES], val_task[target])],
                verbose=False,
            )
            pred = model.predict(te_task[FEATURES])
            result = metrics(te_task[target], pred, "soh")
            result.update({
                "model": "XGBoost",
                "task": "soh",
                "status": "EVALUATED",
                "target_units": "% SOH",
                "target_definition": next(iter(definitions)),
                "train_rows": int(len(tr_task)),
                "validation_rows": int(len(val_task)),
                "test_rows": int(len(te_task)),
                "train_battery_ids": split.train_ids,
                "validation_battery_ids": split.validation_ids,
                "test_battery_ids": split.test_ids,
                "best_iteration": int(model.best_iteration),
            })
            experiments.append(result)
            joblib.dump(
                {
                    "model": model,
                    "feature_order": FEATURES,
                    "preprocessing": "none",
                    "target": "soh_percent",
                    "target_definition": next(iter(definitions)),
                    "training_status": "EVALUATED",
                    "seed": SEED,
                    "train_battery_ids": split.train_ids,
                    "validation_battery_ids": split.validation_ids,
                    "test_battery_ids": split.test_ids,
                },
                MODELS / "xgboost_soh.joblib",
            )
    except ImportError:
        warnings.warn("xgboost is not installed; skipping XGBoost.")

    # Deep sequence models use chronological windows per battery; never mix identities between splits.
    try:
        import torch
        from torch import nn
        torch.manual_seed(SEED)
        torch.set_num_threads(1)
        class Recurrent(nn.Module):
            def __init__(self, kind, n_features, hybrid=False):
                super().__init__(); self.hybrid=hybrid
                if hybrid: self.conv=nn.Sequential(nn.Conv1d(n_features,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,32,3,padding=1),nn.ReLU())
                rnn_input=32 if hybrid else n_features
                self.rnn=(nn.LSTM if kind=='LSTM' else nn.GRU)(rnn_input,32,batch_first=True)
                self.head=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
            def forward(self,x):
                if self.hybrid: x=self.conv(x.transpose(1,2)).transpose(1,2)
                z,_=self.rnn(x); return self.head(z[:,-1,:]).squeeze(-1)
        def fit_seq(name):
            target = "soh_percent"
            train_rows = train.dropna(subset=[target])
            validation_rows = validation.dropna(subset=[target])
            test_rows = test.dropna(subset=[target])
            X_train, y_train = sequence_windows(
                train_rows, target, FEATURES, x_scaler, sequence_length=10
            )
            X_validation, y_validation = sequence_windows(
                validation_rows, target, FEATURES, x_scaler, sequence_length=10
            )
            X_test, y_test = sequence_windows(
                test_rows, target, FEATURES, x_scaler, sequence_length=10
            )
            if len(X_train) < 10 or len(X_validation) < 1 or len(X_test) < 1:
                print(
                    f"Skipping {name} SOH: too few chronological windows "
                    f"(train={len(X_train)}, validation={len(X_validation)}, "
                    f"test={len(X_test)})."
                )
                return
            model = Recurrent(
                name,
                len(FEATURES),
                hybrid=(name == "Hybrid CNN-LSTM"),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            loss_function = nn.MSELoss()
            train_x = torch.tensor(X_train)
            train_y = torch.tensor(y_train)
            validation_x = torch.tensor(X_validation)
            validation_y = torch.tensor(y_validation)
            best_validation_loss = float("inf")
            best_state = None
            patience = 6
            epochs_without_improvement = 0
            best_epoch = 0

            for epoch in range(35):
                model.train()
                perm = torch.randperm(len(train_x))
                for ix in perm.split(64):
                    optimizer.zero_grad()
                    loss = loss_function(model(train_x[ix]), train_y[ix])
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    validation_loss = float(
                        loss_function(model(validation_x), validation_y).item()
                    )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break

            if best_state is None:
                raise RuntimeError(f"{name} did not produce a validation checkpoint.")
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                prediction = model(torch.tensor(X_test)).numpy()
            result = metrics(y_test, prediction, "soh")
            result.update({
                "model": name,
                "task": "soh",
                "status": "EVALUATED",
                "target_units": "% SOH",
                "target_definition": next(iter(definitions)),
                "train_rows": int(len(y_train)),
                "validation_rows": int(len(y_validation)),
                "test_rows": int(len(y_test)),
                "train_battery_ids": split.train_ids,
                "validation_battery_ids": split.validation_ids,
                "test_battery_ids": split.test_ids,
                "window_length": 10,
                "best_epoch": best_epoch,
                "best_validation_mse": best_validation_loss,
            })
            experiments.append(result)
            torch.save({
                "state_dict": model.state_dict(),
                "model": name,
                "task": "soh",
                "feature_order": FEATURES,
                "window_length": 10,
                "preprocessing": "StandardScaler fit on training identities",
                "feature_scaler": x_scaler,
                "target_definition": next(iter(definitions)),
                "training_status": "EVALUATED",
                "seed": SEED,
                "train_battery_ids": split.train_ids,
                "validation_battery_ids": split.validation_ids,
                "test_battery_ids": split.test_ids,
            }, MODELS / f"{name.lower().replace(' ', '_').replace('-', '_')}_soh.pt")
        for model_name in ["LSTM","GRU","Hybrid CNN-LSTM"]:
            fit_seq(model_name)
    except ImportError:
        warnings.warn("PyTorch is not installed; sequence models were skipped. Install torch to run LSTM, GRU and Hybrid CNN-LSTM.")

    out={
        "dataset_status": "evaluated" if experiments else "no successful experiments",
        "source_file": "data/processed/battery_measurements.csv",
        "target_definitions": {"soh_percent": next(iter(definitions))},
        "split": {
            "method": "battery identity train/validation/test holdout",
            "seed": SEED,
            "train_battery_ids": split.train_ids,
            "validation_battery_ids": split.validation_ids,
            "test_battery_ids": split.test_ids,
            "identity_overlap": [],
        },
        "feature_columns": FEATURES,
        "experiments": experiments,
        "limitations": [
            "SOH training remains blocked until the target and protocol are justified.",
            "Battery-cell data are not vehicle-range data.",
            "MAPE excludes targets at or below the recorded near-zero threshold.",
        ],
    }
    (RESULTS/"model_metrics.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(f"Wrote {RESULTS/'model_metrics.json'} with {len(experiments)} experiment rows.")

if __name__=='__main__': main()
