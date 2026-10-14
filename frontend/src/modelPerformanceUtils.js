/**
 * VoltWise Model Performance Utilities
 *
 * Provides benchmark metrics constants, multi-model evaluation helpers,
 * input validation, metric-based winner selection, and prediction spread calculations.
 */

export const HELD_OUT_TRIP_BENCHMARKS = [
  {
    model: 'XGBoost',
    modelKey: 'xgboost',
    architecture: 'Tabular 30s Statistical Aggregates',
    rmse: 1.488,
    mae: 1.285,
    r2: -0.249,
    mape: 130.56,
    color: '#059669', // Emerald
    notes: 'Best tree model; lowest overall validation RMSE',
  },
  {
    model: 'GRU',
    modelKey: 'gru',
    architecture: '2-Layer Gated Recurrent Unit (30s sequence)',
    rmse: 1.584,
    mae: 0.905,
    r2: -0.416,
    mape: 45.60,
    color: '#2563EB', // Blue
    notes: 'Lowest absolute error (MAE 0.905 km) and lowest MAPE (45.60%)',
  },
  {
    model: 'LSTM',
    modelKey: 'lstm',
    architecture: '2-Layer Long Short-Term Memory (30s sequence)',
    rmse: 1.541,
    mae: 1.370,
    r2: -0.339,
    mape: 134.87,
    color: '#7C3AED', // Purple
    notes: 'Recurrent sequence network on 7-channel telemetry',
  },
  {
    model: 'Hybrid CNN-LSTM',
    modelKey: 'hybrid',
    architecture: '1D-CNN temporal feature extractor + LSTM layer',
    rmse: 1.572,
    mae: 1.392,
    r2: -0.393,
    mape: 141.83,
    color: '#D97706', // Amber
    notes: 'Convolutional-recurrent hybrid architecture',
  },
  {
    model: 'Training-Mean Baseline',
    modelKey: 'baseline',
    architecture: 'Constant Training-Set Mean (2.601 km)',
    rmse: 1.499,
    mae: 1.291,
    r2: -0.266,
    mape: 131.90,
    color: '#64748B', // Slate
    notes: 'Non-learning reference baseline for completed distance',
  },
];

/**
 * Validates inputs for trip distance inference.
 */
export function validateTripDistanceInputs({ windowDistanceKm, speedMean, powerMean }) {
  const errors = {};

  const dist = Number(windowDistanceKm);
  if (windowDistanceKm === '' || windowDistanceKm === null || isNaN(dist) || dist <= 0 || dist > 5.0) {
    errors.windowDistanceKm = 'Initial 30s distance must be a positive number up to 5.00 km (typically 0.1–1.0 km).';
  }

  const speed = Number(speedMean);
  if (speedMean === '' || speedMean === null || isNaN(speed) || speed < 1 || speed > 160) {
    errors.speedMean = 'Mean speed must be between 1 km/h and 160 km/h.';
  }

  const power = Number(powerMean);
  if (powerMean === '' || powerMean === null || isNaN(power) || power < -50 || power > 200) {
    errors.powerMean = 'Mean power must be between -50 kW (regen) and 200 kW (discharge).';
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}

/**
 * Determines the benchmark winner based on the selected metric ('rmse' or 'mae').
 *
 * Benchmark facts:
 * - Lowest RMSE: XGBoost (1.488 km)
 * - Lowest MAE: GRU (0.905 km)
 */
export function determineBestModel(benchmarks = HELD_OUT_TRIP_BENCHMARKS, metric = 'rmse') {
  if (!Array.isArray(benchmarks) || benchmarks.length === 0) {
    return null;
  }

  const valid = benchmarks.filter((m) => Number.isFinite(m[metric]) && m[metric] > 0);
  if (valid.length === 0) return null;

  // Minimum value is the winner for error metrics (RMSE, MAE)
  const winner = valid.reduce((best, cur) => (cur[metric] < best[metric] ? cur : best), valid[0]);

  return {
    winnerModel: winner.model,
    metricName: metric.toUpperCase(),
    metricValue: winner[metric],
    explanation:
      metric.toLowerCase() === 'rmse'
        ? `${winner.model} achieved the lowest Root Mean Squared Error (${winner.rmse.toFixed(3)} km) on held-out Vehicle 10.`
        : `${winner.model} achieved the lowest Mean Absolute Error (${winner.mae.toFixed(3)} km) and lowest MAPE (${winner.mape?.toFixed(1) || '45.6'}%) on held-out Vehicle 10.`,
    winner,
  };
}

/**
 * Calculates prediction spread across successful model inferences.
 */
export function calculatePredictionSpread(predictions) {
  if (!Array.isArray(predictions) || predictions.length === 0) {
    return {
      min: null,
      max: null,
      mean: null,
      spread: null,
      successfulCount: 0,
    };
  }

  const valid = predictions
    .map((p) => (typeof p.predictedKm === 'number' && Number.isFinite(p.predictedKm) ? p.predictedKm : null))
    .filter((v) => v !== null && v >= 0);

  if (valid.length === 0) {
    return {
      min: null,
      max: null,
      mean: null,
      spread: null,
      successfulCount: 0,
    };
  }

  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const mean = valid.reduce((sum, v) => sum + v, 0) / valid.length;
  const spread = max - min;

  return {
    min: Math.round(min * 1000) / 1000,
    max: Math.round(max * 1000) / 1000,
    mean: Math.round(mean * 1000) / 1000,
    spread: Math.round(spread * 1000) / 1000,
    successfulCount: valid.length,
  };
}
