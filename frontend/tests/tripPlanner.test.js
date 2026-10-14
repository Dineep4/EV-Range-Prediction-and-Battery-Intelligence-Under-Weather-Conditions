import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildRouteResult,
  formatRouteDuration,
  handleTripFormSubmit,
  recommendPotentialChargeStops,
  resolveEpaVehicleSelection,
  sampleRoutePoints,
  validateTripPlanInputs,
} from '../src/tripPlanner.js';

const vehicle = {
  source_record_id: 18291,
  label: '2002 Ford Explorer USPS Electric',
  features: {
    year: 2002,
    city_e_kwh_100_miles: 75,
    highway_e_kwh_100_miles: 102,
    combined_e_kwh_100_miles: 87,
    charge_240_hours: 0,
    drive: '2-Wheel Drive',
    vehicle_class: 'Sport Utility Vehicle - 2WD',
  },
};

test('resolves a selected EPA record by its catalogue ID and exact label', () => {
  assert.equal(
    resolveEpaVehicleSelection(`${vehicle.label} · record 18291`, [vehicle]),
    vehicle,
  );
});

test('rejects invalid or mismatched EPA catalogue selections', () => {
  assert.equal(resolveEpaVehicleSelection('2002 Ford Explorer', [vehicle]), null);
  assert.equal(
    resolveEpaVehicleSelection('2002 Ford Explorer USPS Electric · record 9', [vehicle]),
    null,
  );
  assert.equal(
    resolveEpaVehicleSelection('2002 Ford Explorer · record 18291', [vehicle]),
    null,
  );
});

test('formats routed duration in minutes and hours with exact-hour handling', () => {
  assert.equal(formatRouteDuration(45 * 60), '45 min');
  assert.equal(formatRouteDuration(60 * 60), '1 hr');
  assert.equal(formatRouteDuration(75 * 60), '1 hr 15 min');
  assert.equal(formatRouteDuration(120 * 60), '2 hr');
  assert.equal(formatRouteDuration(130 * 60), '2 hr 10 min');
  assert.equal(formatRouteDuration(-1), 'Unavailable');
});

test('samples long routes with both endpoints and a bounded request count', () => {
  const points = Array.from({ length: 100 }, (_, index) => [index, index]);
  const samples = sampleRoutePoints(points, 8);
  assert.equal(samples.length, 8);
  assert.deepEqual(samples[0], points[0]);
  assert.deepEqual(samples.at(-1), points.at(-1));
});

test('recommends only mapped stations that are approximate reachable route candidates', () => {
  const stations = [
    { id: 1, routeProgressKm: 45, distanceFromRouteKm: 1 },
    { id: 2, routeProgressKm: 75, distanceFromRouteKm: 0.5 },
    { id: 3, routeProgressKm: 20, distanceFromRouteKm: 5 },
  ];
  assert.deepEqual(
    recommendPotentialChargeStops(stations, 60).map((item) => item.id),
    [1],
  );
  assert.deepEqual(recommendPotentialChargeStops(stations, 0), []);
});

test('handleTripFormSubmit prevents native browser submit and invokes planner handler', () => {
  let prevented = false;
  let executed = false;
  const mockEvent = {
    preventDefault() {
      prevented = true;
    },
  };
  handleTripFormSubmit(mockEvent, () => {
    executed = true;
  });
  assert.equal(prevented, true, 'Default form submission must be prevented to avoid page reload');
  assert.equal(executed, true, 'Planner callback must be executed');

  // Also handles click without event safely
  let executedNoEvent = false;
  handleTripFormSubmit(null, () => {
    executedNoEvent = true;
  });
  assert.equal(executedNoEvent, true);
});

test('validateTripPlanInputs requires non-empty and distinct endpoints', () => {
  assert.throws(
    () => validateTripPlanInputs({ start: '', destination: 'Charminar' }),
    /Enter both a starting point and a destination/,
  );
  assert.throws(
    () => validateTripPlanInputs({ start: 'Hyderabad', destination: '   ' }),
    /Enter both a starting point and a destination/,
  );
  assert.throws(
    () => validateTripPlanInputs({ start: 'Hyderabad', destination: 'hyderabad' }),
    /Choose different starting and destination locations/,
  );

  const clean = validateTripPlanInputs({
    start: '  VNR VJIET, Hyderabad  ',
    destination: '  Charminar, Hyderabad  ',
  });
  assert.equal(clean.start, 'VNR VJIET, Hyderabad');
  assert.equal(clean.destination, 'Charminar, Hyderabad');
});

test('buildRouteResult constructs complete route metrics and range comparisons', () => {
  const mockRouteData = {
    start: { latitude: 17.538, longitude: 78.385, label: 'VNR VJIET' },
    destination: { latitude: 17.361, longitude: 78.474, label: 'Charminar' },
    distance_m: 25300,
    duration_s: 2050,
    geometry: [
      { latitude: 17.538, longitude: 78.385 },
      { latitude: 17.361, longitude: 78.474 },
    ],
    source: 'OSRM routing',
  };
  const mockPhysics = {
    estimatedRangeKm: 275.4,
    usableEnergyKwh: 45.4,
    unadjustedRangeKm: 285.0,
  };

  const result = buildRouteResult({
    routeData: mockRouteData,
    physics: mockPhysics,
    epaResult: { predicted_epa_range_km: 260 },
    tripWeatherFactor: 0.95,
  });

  assert.equal(result.distanceKm, 25.3);
  assert.equal(result.durationSeconds, 2050);
  assert.equal(result.availableRangeKm, 275.4);
  assert.equal(result.weatherAdjusted, true);
  assert.equal(result.epaPrediction.predicted_epa_range_km, 260);
  assert.deepEqual(result.start, [17.538, 78.385]);
});

test('buildRouteResult rejects incomplete routing geometry or non-positive distance', () => {
  assert.throws(
    () => buildRouteResult({
      routeData: { geometry: [], distance_m: 0, duration_s: 100 },
      physics: { estimatedRangeKm: 200 },
    }),
    /incomplete route distance, duration, or geometry/,
  );

  assert.throws(
    () => buildRouteResult({
      routeData: {
        geometry: [{ latitude: 17.5, longitude: 78.3 }],
        distance_m: 5000,
        duration_s: 300,
      },
      physics: { estimatedRangeKm: 200 },
    }),
    /incomplete route distance, duration, or geometry/,
  );
});
