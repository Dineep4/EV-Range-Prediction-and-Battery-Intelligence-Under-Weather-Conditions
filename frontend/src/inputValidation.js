/**
 * VoltWise — Input Validation Utility
 *
 * Implements robust validation rules for Dashboard and Trip Planner numerical and route inputs.
 * - Battery capacity: > 0 kWh and <= 300 kWh
 * - State of Charge (SOC): 0% to 100% inclusive
 * - Energy consumption: > 0 Wh/km and <= 1000 Wh/km (prevents division-by-zero)
 * - Planning capacity adjustment (SOH): > 0% and <= 100%
 * - Route endpoints: Non-empty, >= 2 characters, distinct
 */

export function validateBatteryCapacity(value) {
  if (value === '' || value === null || value === undefined) {
    return 'Battery capacity is required.';
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return 'Battery capacity must be a valid number.';
  }
  if (num <= 0) {
    return 'Battery capacity must be greater than 0 kWh.';
  }
  if (num > 300) {
    return 'Battery capacity cannot exceed 300 kWh.';
  }
  return null;
}

export function validateSOC(value) {
  if (value === '' || value === null || value === undefined) {
    return 'State of charge is required.';
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return 'State of charge must be a valid number.';
  }
  if (num < 0 || num > 100) {
    return 'SOC must be between 0% and 100%.';
  }
  return null;
}

export function validateConsumption(value) {
  if (value === '' || value === null || value === undefined) {
    return 'Energy consumption is required.';
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return 'Energy consumption must be a valid number.';
  }
  if (num <= 0) {
    return 'Energy consumption must be greater than 0 Wh/km.';
  }
  if (num > 1000) {
    return 'Energy consumption cannot exceed 1000 Wh/km.';
  }
  return null;
}

export function validateSOH(value) {
  if (value === '' || value === null || value === undefined) {
    return 'Planning adjustment is required.';
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return 'Planning adjustment must be a valid number.';
  }
  if (num <= 0) {
    return 'Planning adjustment must be greater than 0%.';
  }
  if (num > 100) {
    return 'Planning adjustment cannot exceed 100%.';
  }
  return null;
}

export function validateRouteLocations(start, destination) {
  const errors = {};
  const cleanStart = String(start || '').trim();
  const cleanDest = String(destination || '').trim();

  if (!cleanStart) {
    errors.start = 'Please enter a valid starting point.';
  } else if (cleanStart.length < 2) {
    errors.start = 'Starting point must be at least 2 characters.';
  }

  if (!cleanDest) {
    errors.destination = 'Please enter a valid destination.';
  } else if (cleanDest.length < 2) {
    errors.destination = 'Destination must be at least 2 characters.';
  }

  if (cleanStart && cleanDest && cleanStart.toLowerCase() === cleanDest.toLowerCase()) {
    errors.destination = 'Please enter distinct starting and destination locations.';
  }

  return errors;
}

export function validateAllPlanningInputs({ start, destination, capacity, soc, soh, consumption }) {
  const errors = {};
  const routeErrors = validateRouteLocations(start, destination);
  if (routeErrors.start) errors.start = routeErrors.start;
  if (routeErrors.destination) errors.destination = routeErrors.destination;

  const capErr = validateBatteryCapacity(capacity);
  if (capErr) errors.capacity = capErr;

  const socErr = validateSOC(soc);
  if (socErr) errors.soc = socErr;

  const sohErr = validateSOH(soh);
  if (sohErr) errors.soh = sohErr;

  const consErr = validateConsumption(consumption);
  if (consErr) errors.consumption = consErr;

  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
}
