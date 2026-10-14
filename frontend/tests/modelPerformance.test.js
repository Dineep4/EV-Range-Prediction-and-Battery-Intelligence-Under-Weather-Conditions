import test from 'node:test';
import assert from 'node:assert/strict';

import {
  HELD_OUT_TRIP_BENCHMARKS,
  validateTripDistanceInputs,
  determineBestModel,
  calculatePredictionSpread,
} from '../src/modelPerformanceUtils.js';

test('1. Validates trip distance inputs correctly', () => {
  // Valid inputs
  const valid = validateTripDistanceInputs({
    windowDistanceKm: 0.28,
    speedMean: 35,
    powerMean: 10,
  });
  assert.equal(valid.isValid, true);
  assert.deepEqual(valid.errors, {});

  // Invalid inputs (negative distance, speed out of bounds, power out of bounds)
  const invalid = validateTripDistanceInputs({
    windowDistanceKm: -0.5,
    speedMean: 200,
    powerMean: -100,
  });
  assert.equal(invalid.isValid, false);
  assert.ok(invalid.errors.windowDistanceKm);
  assert.ok(invalid.errors.speedMean);
  assert.ok(invalid.errors.powerMean);

  // Missing values
  const missing = validateTripDistanceInputs({
    windowDistanceKm: '',
    speedMean: null,
    powerMean: undefined,
  });
  assert.equal(missing.isValid, false);
});

test('2. Identifies XGBoost as winner when RMSE is selected', () => {
  const result = determineBestModel(HELD_OUT_TRIP_BENCHMARKS, 'rmse');
  assert.equal(result.winnerModel, 'XGBoost');
  assert.equal(result.metricName, 'RMSE');
  assert.equal(result.metricValue, 1.488);
  assert.ok(result.explanation.includes('XGBoost achieved the lowest Root Mean Squared Error'));
});

test('3. Identifies GRU as winner when MAE is selected', () => {
  const result = determineBestModel(HELD_OUT_TRIP_BENCHMARKS, 'mae');
  assert.equal(result.winnerModel, 'GRU');
  assert.equal(result.metricName, 'MAE');
  assert.equal(result.metricValue, 0.905);
  assert.ok(result.explanation.includes('GRU achieved the lowest Mean Absolute Error'));
});

test('4. Correctly calculates prediction spread across models', () => {
  const mockPredictions = [
    { model: 'XGBoost', predictedKm: 2.586 },
    { model: 'GRU', predictedKm: 0.683 },
    { model: 'LSTM', predictedKm: 2.596 },
    { model: 'Hybrid CNN-LSTM', predictedKm: 2.752 },
    { model: 'Baseline', predictedKm: 2.601 },
  ];

  const spread = calculatePredictionSpread(mockPredictions);
  assert.equal(spread.successfulCount, 5);
  assert.equal(spread.min, 0.683);
  assert.equal(spread.max, 2.752);
  assert.equal(spread.spread, 2.069);
  assert.ok(spread.mean > 2.0 && spread.mean < 2.5);
});

test('5. Handles partial model failures gracefully in spread calculation', () => {
  const partialPredictions = [
    { model: 'XGBoost', predictedKm: 2.586 },
    { model: 'GRU', predictedKm: null, error: 'Network timeout' },
    { model: 'LSTM', predictedKm: 2.596 },
  ];

  const spread = calculatePredictionSpread(partialPredictions);
  assert.equal(spread.successfulCount, 2);
  assert.equal(spread.min, 2.586);
  assert.equal(spread.max, 2.596);
  assert.equal(spread.spread, 0.01);
});

test('6. Returns empty spread when no predictions are successful', () => {
  const failedPredictions = [
    { model: 'XGBoost', predictedKm: null, error: 'Server error' },
  ];

  const spread = calculatePredictionSpread(failedPredictions);
  assert.equal(spread.successfulCount, 0);
  assert.equal(spread.min, null);
  assert.equal(spread.max, null);
  assert.equal(spread.spread, null);
});
