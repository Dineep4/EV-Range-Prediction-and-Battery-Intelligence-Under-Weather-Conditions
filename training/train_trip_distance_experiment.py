"""Train and evaluate four model families for initial-window EV trip-distance prediction.

Models evaluated:
1. Training-mean baseline
2. XGBoost Regressor
3. LSTM Sequence Regressor
4. GRU Sequence Regressor
5. Hybrid CNN-LSTM Sequence Regressor

All models receive the identical past-only 30-second observation window and are
evaluated on the identical held-out test vehicle partition (Vehicle 10, 12 trips).
No future telemetry or target leakage information is permitted.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "ved"
DATASET_CSV = DATA_DIR / "ved_trip_distance_dataset.csv"
SEQUENCES_NPZ = DATA_DIR / "ved_trip_distance_sequences.npz"
SPLIT_MANIFEST_JSON = DATA_DIR / "ved_trip_distance_split_manifest.json"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42

TABULAR_FEATURES = [
    "window_duration_s",
    "window_distance_km",
    "speed_mean",
    "speed_max",
    "speed_min",
    "speed_std",
    "power_mean",
    "power_max",
    "power_min",
    "current_mean",
    "voltage_mean",
    "soc_start",
    "soc_current",
    "soc_delta",
    "temperature_mean",
]

TARGET_COLUMN = "total_trip_distance_km"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_name: str = "trip_distance_km") -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics require non-empty matching arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Cannot calculate metrics on non-finite values.")

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))

    # Safe MAPE with near-zero handling (> 0.01 km)
    valid_mask = np.abs(actual) > 0.01
    mape = (
        float(np.mean(np.abs((actual[valid_mask] - predicted[valid_mask]) / actual[valid_mask])) * 100.0)
        if valid_mask.any()
        else None
    )

    r2 = float(r2_score(actual, predicted)) if (len(actual) >= 2 and np.var(actual) > 1e-12) else None

    return {
        "task": task_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "mape_valid_count": int(valid_mask.sum()),
        "prediction_count": int(len(actual)),
    }


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, T, F) -> transpose to (B, F, T) for 1D conv
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(conv_out)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


def train_neural_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 80,
    patience: int = 15,
    lr: float = 0.005,
    seed: int = SEED,
) -> tuple[nn.Module, int, float]:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_X_t = torch.tensor(X_train, dtype=torch.float32)
    train_y_t = torch.tensor(y_train, dtype=torch.float32)
    val_X_t = torch.tensor(X_val, dtype=torch.float32)
    val_y_t = torch.tensor(y_val, dtype=torch.float32)

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_X_t))
        for batch_indices in permutation.split(16):
            optimizer.zero_grad()
            preds = model(train_X_t[batch_indices])
            loss = loss_fn(preds, train_y_t[batch_indices])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_X_t), val_y_t).item())

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    return model, best_epoch, best_loss


def run_experiments():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_CSV.exists() or not SEQUENCES_NPZ.exists() or not SPLIT_MANIFEST_JSON.exists():
        raise FileNotFoundError("Dataset files not found. Run scripts/prepare_trip_distance_dataset.py first.")

    dataset_df = pd.read_csv(DATASET_CSV)
    seq_data = np.load(SEQUENCES_NPZ)
    sequences = seq_data["sequences"]  # (51, 30, 7)
    trip_keys = list(seq_data["trip_keys"])
    manifest = json.loads(SPLIT_MANIFEST_JSON.read_text(encoding="utf-8"))

    train_keys = set(manifest["train_trip_keys"])
    val_keys = set(manifest["validation_trip_keys"])
    test_keys = set(manifest["test_trip_keys"])

    print(f"Dataset loaded: {len(dataset_df)} trips total.")
    print(f"Partitions: Train={len(train_keys)}, Val={len(val_keys)}, Test={len(test_keys)}")

    # Index mapping
    key_to_idx = {k: i for i, k in enumerate(trip_keys)}
    train_idx = [key_to_idx[k] for k in manifest["train_trip_keys"]]
    val_idx = [key_to_idx[k] for k in manifest["validation_trip_keys"]]
    test_idx = [key_to_idx[k] for k in manifest["test_trip_keys"]]

    # Tabular frames
    train_df = dataset_df.iloc[train_idx].copy()
    val_df = dataset_df.iloc[val_idx].copy()
    test_df = dataset_df.iloc[test_idx].copy()

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    # 1. Training-Mean Baseline
    mean_val = float(np.mean(y_train))
    baseline_preds = np.full_like(y_test, fill_value=mean_val)
    baseline_metrics = calculate_metrics(y_test, baseline_preds)
    baseline_metrics.update({
        "model": "Training-Mean Baseline",
        "status": "EVALUATED",
        "description": f"Predicts training set mean distance ({mean_val:.3f} km)",
        "train_mean_km": mean_val,
        "seed": SEED,
    })

    # 2. XGBoost Regressor
    X_train_tab = train_df[TABULAR_FEATURES].to_numpy(dtype=float)
    X_val_tab = val_df[TABULAR_FEATURES].to_numpy(dtype=float)
    X_test_tab = test_df[TABULAR_FEATURES].to_numpy(dtype=float)

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=2,
        early_stopping_rounds=20,
    )
    xgb_model.fit(
        X_train_tab,
        y_train,
        eval_set=[(X_val_tab, y_val)],
        verbose=False,
    )
    xgb_preds = xgb_model.predict(X_test_tab)
    xgb_metrics = calculate_metrics(y_test, xgb_preds)
    xgb_metrics.update({
        "model": "XGBoost",
        "status": "EVALUATED",
        "best_iteration": int(xgb_model.best_iteration),
        "feature_count": len(TABULAR_FEATURES),
        "seed": SEED,
    })

    joblib.dump(
        {
            "model": xgb_model,
            "feature_names": TABULAR_FEATURES,
            "target": TARGET_COLUMN,
            "target_units": "km",
            "seed": SEED,
            "best_iteration": int(xgb_model.best_iteration),
            "training_status": "EVALUATED",
        },
        MODELS_DIR / "trip_distance_xgboost.joblib",
    )

    # Sequence model preparation: Standardize features using train split ONLY
    # sequences shape: (N, 30, 7)
    train_seqs = sequences[train_idx]  # (31, 30, 7)
    val_seqs = sequences[val_idx]      # (8, 30, 7)
    test_seqs = sequences[test_idx]    # (12, 30, 7)

    N_tr, T, F = train_seqs.shape
    scaler = StandardScaler()
    scaler.fit(train_seqs.reshape(-1, F))

    X_train_seq = scaler.transform(train_seqs.reshape(-1, F)).reshape(N_tr, T, F)
    X_val_seq = scaler.transform(val_seqs.reshape(-1, F)).reshape(len(val_seqs), T, F)
    X_test_seq = scaler.transform(test_seqs.reshape(-1, F)).reshape(len(test_seqs), T, F)

    # 3. LSTM
    lstm_net = LSTMModel(input_dim=F, hidden_dim=32, num_layers=1)
    lstm_net, lstm_epoch, lstm_val_loss = train_neural_model(
        lstm_net, X_train_seq, y_train, X_val_seq, y_val, max_epochs=80, patience=15, lr=0.005, seed=SEED
    )
    with torch.no_grad():
        lstm_preds = lstm_net(torch.tensor(X_test_seq, dtype=torch.float32)).numpy()
    lstm_metrics = calculate_metrics(y_test, lstm_preds)
    lstm_metrics.update({
        "model": "LSTM",
        "status": "EVALUATED",
        "best_epoch": lstm_epoch,
        "best_val_mse": float(lstm_val_loss),
        "seed": SEED,
    })
    torch.save({
        "state_dict": lstm_net.state_dict(),
        "model": "LSTM",
        "input_dim": F,
        "hidden_dim": 32,
        "scaler": scaler,
        "seed": SEED,
    }, MODELS_DIR / "trip_distance_lstm.pt")

    # 4. GRU
    gru_net = GRUModel(input_dim=F, hidden_dim=32, num_layers=1)
    gru_net, gru_epoch, gru_val_loss = train_neural_model(
        gru_net, X_train_seq, y_train, X_val_seq, y_val, max_epochs=80, patience=15, lr=0.005, seed=SEED
    )
    with torch.no_grad():
        gru_preds = gru_net(torch.tensor(X_test_seq, dtype=torch.float32)).numpy()
    gru_metrics = calculate_metrics(y_test, gru_preds)
    gru_metrics.update({
        "model": "GRU",
        "status": "EVALUATED",
        "best_epoch": gru_epoch,
        "best_val_mse": float(gru_val_loss),
        "seed": SEED,
    })
    torch.save({
        "state_dict": gru_net.state_dict(),
        "model": "GRU",
        "input_dim": F,
        "hidden_dim": 32,
        "scaler": scaler,
        "seed": SEED,
    }, MODELS_DIR / "trip_distance_gru.pt")

    # 5. Hybrid CNN-LSTM
    cnn_lstm_net = CNNLSTMModel(input_dim=F, hidden_dim=32)
    cnn_lstm_net, cnn_lstm_epoch, cnn_lstm_val_loss = train_neural_model(
        cnn_lstm_net, X_train_seq, y_train, X_val_seq, y_val, max_epochs=80, patience=15, lr=0.005, seed=SEED
    )
    with torch.no_grad():
        cnn_lstm_preds = cnn_lstm_net(torch.tensor(X_test_seq, dtype=torch.float32)).numpy()
    cnn_lstm_metrics = calculate_metrics(y_test, cnn_lstm_preds)
    cnn_lstm_metrics.update({
        "model": "Hybrid CNN-LSTM",
        "status": "EVALUATED",
        "best_epoch": cnn_lstm_epoch,
        "best_val_mse": float(cnn_lstm_val_loss),
        "seed": SEED,
    })
    torch.save({
        "state_dict": cnn_lstm_net.state_dict(),
        "model": "Hybrid CNN-LSTM",
        "input_dim": F,
        "hidden_dim": 32,
        "scaler": scaler,
        "seed": SEED,
    }, MODELS_DIR / "trip_distance_cnn_lstm.pt")

    # Save test predictions CSV
    test_results_df = pd.DataFrame({
        "trip_key": test_df["trip_key"].values,
        "vehicle_id": test_df["vehicle_id"].values,
        "trip_id": test_df["trip_id"].values,
        "actual_distance_km": y_test,
        "pred_baseline_mean_km": baseline_preds,
        "pred_xgboost_km": xgb_preds,
        "pred_lstm_km": lstm_preds,
        "pred_gru_km": gru_preds,
        "pred_cnn_lstm_km": cnn_lstm_preds,
    })
    test_results_df.to_csv(RESULTS_DIR / "trip_distance_test_predictions.csv", index=False)
    print(f"Wrote test predictions to {RESULTS_DIR / 'trip_distance_test_predictions.csv'}")

    all_experiments = [
        baseline_metrics,
        xgb_metrics,
        lstm_metrics,
        gru_metrics,
        cnn_lstm_metrics,
    ]

    metrics_payload = {
        "task": "VED Initial-Window EV Trip Distance Prediction",
        "target": "total_trip_distance_km",
        "target_units": "km",
        "target_definition": (
            "Total trip distance in kilometres, derived via trapezoidal integration of speed "
            "over the completed trip for pure-EV trips with >= 95% speed-distance coverage. "
            "Estimated at t = 30 seconds using the initial 30-second telemetry window."
        ),
        "observation_window_s": 30.0,
        "split": {
            "method": "Vehicle-grouped holdout: Vehicle 10 test (12 trips), Vehicle 541 & partial 455 val (8 trips), Vehicle 455 train (31 trips)",
            "seed": SEED,
            "train_trips": len(train_idx),
            "validation_trips": len(val_idx),
            "test_trips": len(test_idx),
            "test_vehicle_ids": ["10"],
        },
        "experiments": all_experiments,
        "limitations": [
            "Dataset includes only 3 pure-EV vehicles, all 2013 Nissan Leafs.",
            "Vehicle 10 is the only completely unseen vehicle identity in the held-out test split (12 trips).",
            "Initial 30-second window captures early driving style and speed profile, but cannot anticipate arbitrary route changes or unexpected trip endpoints.",
            "Results are exploratory and should not be presented as a generalized commercial range prediction model.",
        ],
    }

    metrics_file = RESULTS_DIR / "trip_distance_metrics.json"
    metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Wrote metrics report to {metrics_file}")

    print("\n--- Model Comparison Results (Test Set: n=12 trips of Vehicle 10) ---")
    for exp in all_experiments:
        r2_str = f"{exp['r2']:.4f}" if exp['r2'] is not None else "N/A"
        mape_str = f"{exp['mape']:.2f}%" if exp['mape'] is not None else "N/A"
        print(f"  {exp['model']:<25} | RMSE: {exp['rmse']:.3f} km | MAE: {exp['mae']:.3f} km | R2: {r2_str:>7} | MAPE: {mape_str}")


if __name__ == "__main__":
    run_experiments()
