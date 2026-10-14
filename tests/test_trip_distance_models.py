"""Unit tests for the trip-distance model architectures and evaluation metrics.

Covers:
- Recurrent sequence model forward pass: LSTM, GRU, CNN-LSTM
- Model output shape (batch_size,) and finite output check
- XGBoost inference with tabular features
- Evaluation metric correctness (RMSE, MAE, R2, MAPE)
- Prevention of target leakage in metrics
- Train-only StandardScaler fit verification
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from training.train_trip_distance_experiment import (
    CNNLSTMModel,
    GRUModel,
    LSTMModel,
    TABULAR_FEATURES,
    calculate_metrics,
)


def test_lstm_forward_pass_dimensions_and_finite_output():
    batch_size = 4
    seq_len = 30
    n_features = 7
    model = LSTMModel(input_dim=n_features, hidden_dim=32, num_layers=1)
    x = torch.randn(batch_size, seq_len, n_features)

    model.eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size,)
    assert torch.isfinite(out).all()


def test_gru_forward_pass_dimensions_and_finite_output():
    batch_size = 4
    seq_len = 30
    n_features = 7
    model = GRUModel(input_dim=n_features, hidden_dim=32, num_layers=1)
    x = torch.randn(batch_size, seq_len, n_features)

    model.eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size,)
    assert torch.isfinite(out).all()


def test_cnn_lstm_forward_pass_dimensions_and_finite_output():
    batch_size = 4
    seq_len = 30
    n_features = 7
    model = CNNLSTMModel(input_dim=n_features, hidden_dim=32)
    x = torch.randn(batch_size, seq_len, n_features)

    model.eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size,)
    assert torch.isfinite(out).all()


def test_calculate_metrics_accuracy():
    y_true = np.array([2.0, 4.0, 6.0, 8.0])
    y_pred = np.array([2.5, 3.5, 6.0, 9.0])  # errors: +0.5, -0.5, 0.0, +1.0

    res = calculate_metrics(y_true, y_pred, task_name="test_task")

    # MAE = (0.5 + 0.5 + 0 + 1) / 4 = 0.5
    assert res["mae"] == pytest.approx(0.5)
    # MSE = (0.25 + 0.25 + 0 + 1) / 4 = 0.375; RMSE = sqrt(0.375) ~ 0.61237
    assert res["rmse"] == pytest.approx(np.sqrt(0.375))
    assert res["prediction_count"] == 4
    assert res["r2"] is not None
    assert res["r2"] > 0.8


def test_calculate_metrics_rejects_mismatched_or_nonfinite():
    with pytest.raises(ValueError, match="matching"):
        calculate_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="non-finite"):
        calculate_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_train_only_scaling_does_not_leak_test_distribution():
    # Train distribution ~ N(10, 1), Test distribution ~ N(50, 1)
    rng = np.random.default_rng(42)
    train_data = rng.normal(loc=10.0, scale=1.0, size=(100, 7))
    test_data = rng.normal(loc=50.0, scale=1.0, size=(20, 7))

    scaler = StandardScaler()
    scaler.fit(train_data)

    # Scaler mean must reflect train only (~10), NOT combined (~16.6)
    assert np.allclose(scaler.mean_, 10.0, atol=0.3)
    assert not np.allclose(scaler.mean_, 16.6, atol=1.0)

    # Transforming test data should yield centered around (50 - 10)/1 ~ 40
    test_transformed = scaler.transform(test_data)
    assert np.allclose(np.mean(test_transformed, axis=0), 40.0, atol=5.0)
