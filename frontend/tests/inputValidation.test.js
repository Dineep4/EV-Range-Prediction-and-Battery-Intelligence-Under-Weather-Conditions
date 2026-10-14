import test from 'node:test';
import assert from 'node:assert/strict';
import {
  validateBatteryCapacity,
  validateSOC,
  validateConsumption,
  validateSOH,
  validateRouteLocations,
  validateAllPlanningInputs
} from '../src/inputValidation.js';

test('1. validateBatteryCapacity rejects invalid values and accepts valid capacity', () => {
  assert.equal(validateBatteryCapacity(60), null);
  assert.equal(validateBatteryCapacity('75'), null);
  assert.equal(validateBatteryCapacity(0), 'Battery capacity must be greater than 0 kWh.');
  assert.equal(validateBatteryCapacity(-10), 'Battery capacity must be greater than 0 kWh.');
  assert.equal(validateBatteryCapacity(''), 'Battery capacity is required.');
  assert.equal(validateBatteryCapacity('abc'), 'Battery capacity must be a valid number.');
  assert.equal(validateBatteryCapacity(350), 'Battery capacity cannot exceed 300 kWh.');
});

test('2. validateSOC enforces 0% to 100% inclusive range', () => {
  assert.equal(validateSOC(0), null);
  assert.equal(validateSOC(100), null);
  assert.equal(validateSOC('82'), null);
  assert.equal(validateSOC(-1), 'SOC must be between 0% and 100%.');
  assert.equal(validateSOC(101), 'SOC must be between 0% and 100%.');
  assert.equal(validateSOC(''), 'State of charge is required.');
  assert.equal(validateSOC('invalid'), 'State of charge must be a valid number.');
});

test('3. validateConsumption rejects non-positive values to prevent division by zero', () => {
  assert.equal(validateConsumption(165), null);
  assert.equal(validateConsumption('200'), null);
  assert.equal(validateConsumption(0), 'Energy consumption must be greater than 0 Wh/km.');
  assert.equal(validateConsumption(-50), 'Energy consumption must be greater than 0 Wh/km.');
  assert.equal(validateConsumption(1200), 'Energy consumption cannot exceed 1000 Wh/km.');
  assert.equal(validateConsumption(''), 'Energy consumption is required.');
});

test('4. validateSOH enforces valid percentage boundaries', () => {
  assert.equal(validateSOH(92), null);
  assert.equal(validateSOH(100), null);
  assert.equal(validateSOH(0), 'Planning adjustment must be greater than 0%.');
  assert.equal(validateSOH(-5), 'Planning adjustment must be greater than 0%.');
  assert.equal(validateSOH(105), 'Planning adjustment cannot exceed 100%.');
  assert.equal(validateSOH(''), 'Planning adjustment is required.');
});

test('5. validateRouteLocations requires non-empty, distinct endpoints', () => {
  assert.deepEqual(validateRouteLocations('VNR VJIET', 'Charminar'), {});

  const emptyRes = validateRouteLocations('', '');
  assert.equal(emptyRes.start, 'Please enter a valid starting point.');
  assert.equal(emptyRes.destination, 'Please enter a valid destination.');

  const sameRes = validateRouteLocations('Hyderabad', 'Hyderabad');
  assert.equal(sameRes.destination, 'Please enter distinct starting and destination locations.');

  const sameCaseInsensitive = validateRouteLocations('hyderabad', 'HYDERABAD');
  assert.equal(sameCaseInsensitive.destination, 'Please enter distinct starting and destination locations.');
});

test('6. validateAllPlanningInputs aggregates all field errors', () => {
  const valid = validateAllPlanningInputs({
    start: 'Start Point',
    destination: 'End Point',
    capacity: 60,
    soc: 80,
    soh: 90,
    consumption: 160
  });
  assert.equal(valid.isValid, true);
  assert.deepEqual(valid.errors, {});

  const invalid = validateAllPlanningInputs({
    start: '',
    destination: '',
    capacity: 0,
    soc: 150,
    soh: -1,
    consumption: 0
  });
  assert.equal(invalid.isValid, false);
  assert.ok(invalid.errors.start);
  assert.ok(invalid.errors.destination);
  assert.ok(invalid.errors.capacity);
  assert.ok(invalid.errors.soc);
  assert.ok(invalid.errors.soh);
  assert.ok(invalid.errors.consumption);
});
