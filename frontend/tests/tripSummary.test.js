import test from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateDistanceReserve,
  calculateArrivalSocPercent,
  buildTripWeatherPrecautions,
  buildTripChargingAdvice
} from '../src/tripSummaryLogic.js';

test('1. calculateDistanceReserve calculates positive reserve when route is within range', () => {
  const result = calculateDistanceReserve(250, 75.5);
  assert.equal(result.isWithinRange, true);
  assert.equal(result.availableRangeKm, 250);
  assert.equal(result.distanceKm, 75.5);
  assert.equal(result.distanceReserveKm, 174.5);
  assert.equal(result.shortfallKm, 0);
});

test('2. calculateDistanceReserve calculates shortfall when route exceeds range', () => {
  const result = calculateDistanceReserve(200, 260);
  assert.equal(result.isWithinRange, false);
  assert.equal(result.distanceReserveKm, -60);
  assert.equal(result.shortfallKm, 60);
});

test('3. calculateArrivalSocPercent calculates linear arrival SOC properly', () => {
  // 80% starting SOC, 50 km distance, 200 km range -> 75% of battery remains -> 80 * 0.75 = 60%
  const soc1 = calculateArrivalSocPercent(80, 50, 200);
  assert.equal(soc1, 60);

  // When route exceeds range, arrival SOC must be 0% (depleted before arrival)
  const socDepleted = calculateArrivalSocPercent(80, 250, 200);
  assert.equal(socDepleted, 0);

  // Handles invalid or zero values safely
  assert.equal(calculateArrivalSocPercent(80, 0, 0), 0);
  assert.equal(calculateArrivalSocPercent(80, -10, 100), 0);
});

test('4. buildTripWeatherPrecautions handles live weather conditions and unavailable fallback', () => {
  // Unavailable weather
  const unavail = buildTripWeatherPrecautions(null, null, 'unavailable');
  assert.equal(unavail.hasWeather, false);
  assert.equal(unavail.headline, 'Weather observation unavailable');
  assert.equal(unavail.severity, 'neutral');

  // Rainy conditions
  const rain = buildTripWeatherPrecautions({ temperature: 22, precipitation: 2.5, wind: 10 }, null, 'live');
  assert.equal(rain.hasWeather, true);
  assert.equal(rain.severity, 'warning');
  assert.match(rain.headline, /Wet asphalt detected/);

  // Hot conditions (>=30C)
  const hot = buildTripWeatherPrecautions({ temperature: 36, precipitation: 0, wind: 8 }, null, 'live');
  assert.equal(hot.hasWeather, true);
  assert.equal(hot.severity, 'warning');
  assert.match(hot.headline, /High ambient temperature/);

  // Cold conditions (<=10C)
  const cold = buildTripWeatherPrecautions({ temperature: 6, precipitation: 0, wind: 5 }, null, 'live');
  assert.equal(cold.hasWeather, true);
  assert.equal(cold.severity, 'warning');
  assert.match(cold.headline, /Cold ambient conditions/);

  // Mild conditions
  const mild = buildTripWeatherPrecautions({ temperature: 24, precipitation: 0, wind: 10 }, null, 'live');
  assert.equal(mild.hasWeather, true);
  assert.equal(mild.severity, 'positive');
  assert.match(mild.headline, /Mild conditions/);
});

test('5. buildTripChargingAdvice suggests stops for deficit or low margin, and acknowledges comfortable margin', () => {
  // Deficit: requires stop
  const deficit = buildTripChargingAdvice({
    isWithinRange: false,
    distanceReserveKm: -80,
    availableRangeKm: 200,
    potentialStops: [{ name: 'Express Fast Charger', isSimulated: true }]
  });
  assert.equal(deficit.status, 'deficit');
  assert.equal(deficit.needsStop, true);
  assert.match(deficit.headline, /80 km deficit/);
  assert.equal(deficit.hasReachableCandidate, true);
  assert.equal(deficit.candidate.isSimulated, true);

  // Low margin (<15% reserve): prudent stop
  const lowMargin = buildTripChargingAdvice({
    isWithinRange: true,
    distanceReserveKm: 20,
    availableRangeKm: 200,
    potentialStops: []
  });
  assert.equal(lowMargin.status, 'low_margin');
  assert.equal(lowMargin.needsStop, true);
  assert.match(lowMargin.headline, /Low range margin/);

  // Comfortable margin: no intermediate stop needed
  const comfortable = buildTripChargingAdvice({
    isWithinRange: true,
    distanceReserveKm: 120,
    availableRangeKm: 200,
    potentialStops: []
  });
  assert.equal(comfortable.status, 'comfortable');
  assert.equal(comfortable.needsStop, false);
  assert.match(comfortable.headline, /Comfortable range margin/);
});
