/**
 * Battery State of Health (SOH) Utilities and Validation
 */

export const SAMPLES = [
  {
    id: 'early',
    name: 'Early Cycle Baseline (Cycle 10 · ~99% SOH)',
    cycle_index: 10,
    voltage_mean: 3.55,
    current_mean: -1.82,
    temperature_mean: 32.4,
    capacity_ah: 1.84,
    rated_capacity_ah: 2.0,
    notes: 'NASA B0005 baseline cycle; nominal initial capacity.',
  },
  {
    id: 'mid',
    name: 'Mid-Life Aging (Cycle 75 · ~81% SOH)',
    cycle_index: 75,
    voltage_mean: 3.52,
    current_mean: -1.83,
    temperature_mean: 32.8,
    capacity_ah: 1.62,
    rated_capacity_ah: 2.0,
    notes: 'Typical mid-life capacity loss under continuous CC-CV cycling.',
  },
  {
    id: 'late',
    name: 'Late-Stage Aging (Cycle 150 · ~70% SOH)',
    cycle_index: 150,
    voltage_mean: 3.48,
    current_mean: -1.84,
    temperature_mean: 33.5,
    capacity_ah: 1.41,
    rated_capacity_ah: 2.0,
    notes: 'Late-stage aging approaching the 70% first-life retention threshold.',
  },
];

export function validateSOHInputs({
  cycleIndex,
  voltageMean,
  currentMean,
  temperatureMean,
  capacityAh,
  ratedCapacityAh,
}) {
  const errors = {};

  const c = Number(cycleIndex);
  if (cycleIndex === '' || cycleIndex === null || isNaN(c) || !Number.isInteger(c) || c < 1 || c > 500) {
    errors.cycleIndex = 'Cycle count must be an integer between 1 and 500.';
  }

  const v = Number(voltageMean);
  if (voltageMean === '' || voltageMean === null || !Number.isFinite(v) || v < 2.0 || v > 4.5) {
    errors.voltageMean = 'Mean operating voltage must be between 2.00 V and 4.50 V.';
  }

  const i = Number(currentMean);
  if (currentMean === '' || currentMean === null || !Number.isFinite(i) || i < -10.0 || i > 10.0) {
    errors.currentMean = 'Mean load current must be between -10.00 A and 10.00 A.';
  }

  const t = Number(temperatureMean);
  if (temperatureMean === '' || temperatureMean === null || !Number.isFinite(t) || t < 0.0 || t > 70.0) {
    errors.temperatureMean = 'Mean operating temperature must be between 0.0 °C and 70.0 °C.';
  }

  if (capacityAh !== '' && capacityAh !== null && capacityAh !== undefined) {
    const cap = Number(capacityAh);
    if (!Number.isFinite(cap) || cap <= 0.0 || cap > 5.0) {
      errors.capacityAh = 'Measured discharge capacity must be between 0.10 Ah and 5.00 Ah.';
    }
  }

  if (ratedCapacityAh !== '' && ratedCapacityAh !== null && ratedCapacityAh !== undefined) {
    const rated = Number(ratedCapacityAh);
    if (!Number.isFinite(rated) || rated <= 0.0 || rated > 5.0) {
      errors.ratedCapacityAh = 'Rated reference capacity must be between 0.50 Ah and 5.00 Ah.';
    }
  }

  return errors;
}

export function computeEmpiricalUncertaintyInterval(predictedSoh, rmse = 3.498, z = 1.645) {
  const margin = Math.round(z * rmse * 100) / 100;
  const lower = Math.max(0, Math.round((predictedSoh - margin) * 100) / 100);
  const upper = Math.min(120, Math.round((predictedSoh + margin) * 100) / 100);
  return { lowerBound: lower, upperBound: upper, margin };
}
