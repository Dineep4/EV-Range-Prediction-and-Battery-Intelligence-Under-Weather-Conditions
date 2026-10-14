import test from 'node:test';
import assert from 'node:assert/strict';

import {
  generateSimulatedChargingStations,
  resolveSourceLocation,
  calculateDestinationPoint,
  haversineDistanceKm,
  hashCoordinates,
  SIMULATED_STATION_NAMES,
  SIMULATED_DISCLAIMER,
  DEMO_MODE_BANNER,
} from '../src/simulatedChargingStations.js';

test('1. Generates 3-4 simulated stations for a source point in Hyderabad', () => {
  const hyderabadLat = 17.4474;
  const hyderabadLon = 78.3488;

  const stations = generateSimulatedChargingStations({
    sourceLat: hyderabadLat,
    sourceLon: hyderabadLon,
    searchRadiusM: 10000,
  });

  assert.equal(stations.length, 4);
  stations.forEach((station) => {
    assert.equal(station.isSimulated, true);
    assert.equal(station.statusLabel, 'Simulated location');
    assert.equal(station.disclaimer, SIMULATED_DISCLAIMER);
    assert.ok(SIMULATED_STATION_NAMES.includes(station.name));
    assert.ok(Number.isFinite(station.lat));
    assert.ok(Number.isFinite(station.lon));
    // Points should be within search radius and at least 1.5 km away
    assert.ok(station.distanceFromSourceKm >= 1.5, `Distance too small: ${station.distanceFromSourceKm}`);
    assert.ok(station.distanceFromSourceKm <= 12.0, `Distance too large: ${station.distanceFromSourceKm}`);
  });

  // Verify all station names are distinct
  const names = new Set(stations.map((s) => s.name));
  assert.equal(names.size, stations.length, 'Every generated station must have a distinct name');
});

test('2. Generates distinct, geographically plausible stations for a different city (Bengaluru & London)', () => {
  // Bengaluru
  const blrLat = 12.9716;
  const blrLon = 77.5946;
  const blrStations = generateSimulatedChargingStations({
    sourceLat: blrLat,
    sourceLon: blrLon,
    searchRadiusM: 8000,
  });

  assert.equal(blrStations.length, 4);
  blrStations.forEach((station) => {
    assert.ok(station.lat >= 12.8 && station.lat <= 13.1);
    assert.ok(station.lon >= 77.4 && station.lon <= 77.8);
    assert.ok(station.name.length > 0);
  });

  // London (negative lon, high latitude)
  const ldnLat = 51.5074;
  const ldnLon = -0.1278;
  const ldnStations = generateSimulatedChargingStations({
    sourceLat: ldnLat,
    sourceLon: ldnLon,
    searchRadiusM: 12000,
  });

  assert.equal(ldnStations.length, 4);
  ldnStations.forEach((station) => {
    assert.ok(station.lat >= 51.3 && station.lat <= 51.7);
    assert.ok(station.lon >= -0.4 && station.lon <= 0.2);
  });
});

test('3. Generates simulated points with no available trip route (standalone radial search)', () => {
  const stations = generateSimulatedChargingStations({
    sourceLat: 17.385,
    sourceLon: 78.4867,
    searchRadiusM: 6000,
    route: null,
    availableRangeKm: null,
  });

  assert.equal(stations.length, 4);
  stations.forEach((station) => {
    assert.equal(station.recommendationStatus, 'nearby');
    assert.equal(station.routeProgressKm, null);
    assert.ok(station.recommendationReason.includes('Straight-line geodesic distance'));
    assert.ok(station.distanceFromSourceKm > 0);
  });
});

test('4. Range-aware safety reserve: recommends stop around 15-25 km for 40 km range and 50 km route', () => {
  // Mock route from A to B: 50 km total distance
  const routePoints = [
    [17.40, 78.30],
    [17.45, 78.38],
    [17.50, 78.45],
    [17.55, 78.52],
    [17.60, 78.60],
  ];
  const mockRoute = {
    distanceKm: 50.0,
    points: routePoints,
  };
  const availableRangeKm = 40.0;

  const stations = generateSimulatedChargingStations({
    sourceLat: 17.40,
    sourceLon: 78.30,
    route: mockRoute,
    availableRangeKm,
  });

  assert.equal(stations.length, 4);

  // Filter recommended stops
  const recommendedStops = stations.filter((s) => s.recommendationStatus === 'recommended_stop');
  assert.ok(recommendedStops.length >= 1, 'Should recommend at least one safe stop');

  // Verify the recommended stop preserves safety reserve (not at 35-40 km ragged edge)
  recommendedStops.forEach((stop) => {
    assert.ok(
      stop.routeProgressKm <= 28.0,
      `Recommended stop is too close to battery exhaustion: ${stop.routeProgressKm} km on a 40 km range`,
    );
    assert.ok(
      stop.routeProgressKm >= 10.0,
      `Recommended stop is unnecessarily early: ${stop.routeProgressKm} km`,
    );
    assert.ok(stop.recommendationReason.includes('safe battery reserve'));
  });

  // Verify unreachable stop beyond 40 km is marked unreachable
  const unreachableStops = stations.filter((s) => s.routeProgressKm > availableRangeKm);
  unreachableStops.forEach((stop) => {
    assert.equal(stop.isReachable, false);
    assert.equal(stop.recommendationStatus, 'unreachable');
  });
});

test('5. Trip shorter than available battery range: shows stations as optional', () => {
  const mockRoute = {
    distanceKm: 30.0,
    points: [
      [17.40, 78.30],
      [17.48, 78.40],
      [17.55, 78.50],
    ],
  };
  const availableRangeKm = 75.0; // Plenty of range

  const stations = generateSimulatedChargingStations({
    sourceLat: 17.40,
    sourceLon: 78.30,
    route: mockRoute,
    availableRangeKm,
  });

  assert.equal(stations.length, 4);
  stations.forEach((station) => {
    assert.equal(station.recommendationStatus, 'optional');
    assert.equal(station.recommendationBadge, 'Optional Charging Stop');
    assert.ok(station.recommendationReason.includes('within available range'));
  });
});

test('6. Handles invalid or missing coordinates gracefully', () => {
  assert.deepEqual(generateSimulatedChargingStations({ sourceLat: null, sourceLon: null }), []);
  assert.deepEqual(generateSimulatedChargingStations({ sourceLat: 'invalid', sourceLon: 'not-a-num' }), []);
  assert.deepEqual(generateSimulatedChargingStations({ sourceLat: 95.0, sourceLon: 78.0 }), []); // Lat > 90
  assert.deepEqual(generateSimulatedChargingStations({ sourceLat: 17.0, sourceLon: 250.0 }), []); // Lon > 180
});

test('7. Source location priority resolution', () => {
  // Priority 1: Current search coordinates
  const p1 = resolveSourceLocation({
    searchLat: 13.0827,
    searchLon: 80.2707,
    searchLabel: 'Chennai Central',
    routeStart: [17.4, 78.3],
  });
  assert.equal(p1.latitude, 13.0827);
  assert.equal(p1.longitude, 80.2707);
  assert.equal(p1.sourceType, 'search_input');

  // Priority 2: Trip origin when search is empty/invalid
  const p2 = resolveSourceLocation({
    searchLat: '',
    searchLon: '',
    routeStart: [19.076, 72.8777],
    routeStartLabel: 'Mumbai Airport',
  });
  assert.equal(p2.latitude, 19.076);
  assert.equal(p2.longitude, 72.8777);
  assert.equal(p2.sourceType, 'trip_origin');

  // Priority 3: Map selection
  const p3 = resolveSourceLocation({
    searchLat: '',
    searchLon: '',
    routeStart: null,
    mapSelected: { latitude: 28.6139, longitude: 77.209, label: 'New Delhi' },
  });
  assert.equal(p3.latitude, 28.6139);
  assert.equal(p3.longitude, 77.209);
  assert.equal(p3.sourceType, 'map_selection');

  // Priority 4: Clearly labelled default fallback
  const p4 = resolveSourceLocation({
    searchLat: '',
    searchLon: '',
    routeStart: null,
    mapSelected: null,
  });
  assert.equal(p4.sourceType, 'default_fallback');
  assert.ok(p4.label.includes('Default Location'));
});

test('8. Deterministic seed stability: repeated calls at same coordinates return identical coordinates', () => {
  const call1 = generateSimulatedChargingStations({ sourceLat: 17.4474, sourceLon: 78.3488 });
  const call2 = generateSimulatedChargingStations({ sourceLat: 17.4474, sourceLon: 78.3488 });

  assert.equal(call1.length, call2.length);
  for (let i = 0; i < call1.length; i++) {
    assert.equal(call1[i].lat, call2[i].lat);
    assert.equal(call1[i].lon, call2[i].lon);
    assert.equal(call1[i].name, call2[i].name);
    assert.equal(call1[i].chargingPower, call2[i].chargingPower);
  }
});
