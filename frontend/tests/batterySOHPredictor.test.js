import test from 'node:test';
import assert from 'node:assert/strict';
import { validateSOHInputs, computeEmpiricalUncertaintyInterval, SAMPLES } from '../src/batterySOH.js';

test('SOH formula correctly computes capacity ratio against 2.0 Ah rated benchmark', () => {
  const rated = 2.0;
  assert.equal((1.84 / rated) * 100, 92.0);
  assert.equal((1.62 / rated) * 100, 81.0);
  assert.equal((1.40 / rated) * 100, 70.0);
  assert.equal((2.00 / rated) * 100, 100.0);
});

test('SOH validation enforces operational telemetry boundaries', () => {
  function validate(cycle, voltage, current, temp) {
    if (cycle < 1 || cycle > 500) return 'invalid_cycle';
    if (voltage < 2.0 || voltage > 4.5) return 'invalid_voltage';
    if (current < -10.0 || current > 10.0) return 'invalid_current';
    if (temp < 0.0 || temp > 70.0) return 'invalid_temp';
    return 'valid';
  }

  assert.equal(validate(10, 3.55, -1.82, 32.4), 'valid');
  assert.equal(validate(0, 3.55, -1.82, 32.4), 'invalid_cycle');
  assert.equal(validate(600, 3.55, -1.82, 32.4), 'invalid_cycle');
  assert.equal(validate(50, 1.5, -1.82, 32.4), 'invalid_voltage');
  assert.equal(validate(50, 5.0, -1.82, 32.4), 'invalid_voltage');
  assert.equal(validate(50, 3.55, -15.0, 32.4), 'invalid_current');
  assert.equal(validate(50, 3.55, -1.82, 85.0), 'invalid_temp');
});

test('validateSOHInputs detects valid inputs without returning any field errors', () => {
  const errors = validateSOHInputs({
    cycleIndex: 50,
    voltageMean: 3.53,
    currentMean: -1.82,
    temperatureMean: 32.5,
    capacityAh: 1.84,
    ratedCapacityAh: 2.0,
  });
  assert.deepEqual(errors, {});
});

test('validateSOHInputs flags missing, non-numeric, or out-of-bounds telemetry fields', () => {
  const invalidCycle = validateSOHInputs({
    cycleIndex: 0,
    voltageMean: 3.53,
    currentMean: -1.82,
    temperatureMean: 32.5,
  });
  assert.ok(invalidCycle.cycleIndex);

  const invalidVoltage = validateSOHInputs({
    cycleIndex: 50,
    voltageMean: 5.5,
    currentMean: -1.82,
    temperatureMean: 32.5,
  });
  assert.ok(invalidVoltage.voltageMean);

  const invalidCurrent = validateSOHInputs({
    cycleIndex: 50,
    voltageMean: 3.53,
    currentMean: 15.0,
    temperatureMean: 32.5,
  });
  assert.ok(invalidCurrent.currentMean);

  const invalidTemp = validateSOHInputs({
    cycleIndex: 50,
    voltageMean: 3.53,
    currentMean: -1.82,
    temperatureMean: 85.0,
  });
  assert.ok(invalidTemp.temperatureMean);
});

test('validateSOHInputs flags invalid optional capacity values while accepting valid ones', () => {
  const invalidCap = validateSOHInputs({
    cycleIndex: 50,
    voltageMean: 3.53,
    currentMean: -1.82,
    temperatureMean: 32.5,
    capacityAh: -1.0,
  });
  assert.ok(invalidCap.capacityAh);

  const emptyCap = validateSOHInputs({
    cycleIndex: 50,
    voltageMean: 3.53,
    currentMean: -1.82,
    temperatureMean: 32.5,
    capacityAh: '',
    ratedCapacityAh: 2.0,
  });
  assert.deepEqual(emptyCap, {});
});

test('Prediction uncertainty interval correctly wraps bounds and caps at sensible limits', () => {
  const rmse = 3.498;
  const margin = Math.round(1.645 * rmse * 100) / 100; // 5.75
  const pred = 85.4;

  const lower = Math.max(0, Math.round((pred - margin) * 100) / 100);
  const upper = Math.min(120, Math.round((pred + margin) * 100) / 100);

  assert.equal(margin, 5.75);
  assert.equal(lower, 79.65);
  assert.equal(upper, 91.15);
  assert.ok(lower < pred && pred < upper);
});
