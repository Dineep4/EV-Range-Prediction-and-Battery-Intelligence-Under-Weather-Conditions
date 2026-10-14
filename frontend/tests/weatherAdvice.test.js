import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DASHBOARD_WEATHER_THRESHOLDS,
  buildDashboardWeatherEvCareAdvice,
  buildJourneyWeatherPrecautions
} from '../src/weatherAdvice.js';

test('1. DASHBOARD_WEATHER_THRESHOLDS has expected sensible thresholds', () => {
  assert.equal(DASHBOARD_WEATHER_THRESHOLDS.HOT_TEMP_C, 30);
  assert.equal(DASHBOARD_WEATHER_THRESHOLDS.COLD_TEMP_C, 10);
  assert.equal(DASHBOARD_WEATHER_THRESHOLDS.RAIN_PRECIPITATION_MM, 0.1);
  assert.equal(DASHBOARD_WEATHER_THRESHOLDS.STRONG_WIND_KMH, 25);
});

test('2. buildDashboardWeatherEvCareAdvice handles unavailable or null weather gracefully', () => {
  const resNull = buildDashboardWeatherEvCareAdvice(null, 'unavailable');
  assert.equal(resNull.status, 'unavailable');
  assert.equal(resNull.items.length, 1);
  assert.equal(resNull.items[0].id, 'care-unavailable');

  const resLoading = buildDashboardWeatherEvCareAdvice(null, 'loading');
  assert.equal(resLoading.status, 'loading');
  assert.equal(resLoading.items[0].id, 'care-unavailable');
});

test('3. buildDashboardWeatherEvCareAdvice generates hot weather recommendations (>=30°C)', () => {
  const hotWeather = {
    temperature: 34,
    humidity: 40,
    wind: 12,
    precipitation: 0,
    time: '2026-10-13T12:00'
  };
  const res = buildDashboardWeatherEvCareAdvice(hotWeather, 'live');
  assert.equal(res.status, 'live');
  const ids = res.items.map(i => i.id);
  assert.ok(ids.includes('care-hot-parking'), 'Should recommend shaded parking');
  assert.match(res.disclaimer, /not internal battery pack temperature/i);
});

test('4. buildDashboardWeatherEvCareAdvice generates cold weather recommendations (<=10°C)', () => {
  const coldWeather = {
    temperature: 6,
    humidity: 70,
    wind: 10,
    precipitation: 0,
    time: '2026-10-13T06:00'
  };
  const res = buildDashboardWeatherEvCareAdvice(coldWeather, 'live');
  const ids = res.items.map(i => i.id);
  assert.ok(ids.includes('care-cold-range'), 'Should recommend preconditioning and reduced range caution');
});

test('5. buildDashboardWeatherEvCareAdvice generates rainy and windy recommendations', () => {
  const stormWeather = {
    temperature: 20,
    humidity: 90,
    wind: 32,
    precipitation: 4.5,
    time: '2026-10-13T15:00'
  };
  const res = buildDashboardWeatherEvCareAdvice(stormWeather, 'live');
  const ids = res.items.map(i => i.id);
  assert.ok(ids.includes('care-rain-safety'), 'Should recommend wet road precautions');
  assert.ok(ids.includes('care-wind-crosswind'), 'Should recommend crosswind caution');
});

test('6. buildDashboardWeatherEvCareAdvice generates neutral message in mild conditions', () => {
  const mildWeather = {
    temperature: 22,
    humidity: 50,
    wind: 12,
    precipitation: 0,
    time: '2026-10-13T10:00'
  };
  const res = buildDashboardWeatherEvCareAdvice(mildWeather, 'live');
  assert.equal(res.items.length, 1);
  assert.equal(res.items[0].id, 'care-mild-optimal');
  assert.match(res.items[0].text, /No special weather-related precaution/i);
});

test('7. buildJourneyWeatherPrecautions handles endpoint observations for route', () => {
  const route = {
    startLabel: 'Hyderabad Central',
    destinationLabel: 'Warangal Station',
    distanceKm: 145.2,
    durationSeconds: 9800,
    availableRangeKm: 280
  };
  const depWeather = {
    temperature: 32,
    humidity: 45,
    wind: 15,
    precipitation: 0,
    time: '2026-10-13T08:00'
  };
  const destWeather = {
    temperature: 28,
    humidity: 80,
    wind: 28,
    precipitation: 1.2,
    time: '2026-10-13T11:00'
  };

  const analysis = buildJourneyWeatherPrecautions(depWeather, destWeather, route);
  assert.equal(analysis.hasAnyWeather, true);
  const ids = analysis.precautions.map(p => p.id);
  assert.ok(ids.includes('journey-heat'), 'Should trigger heat precaution for departure 32°C');
  assert.ok(ids.includes('journey-rain'), 'Should trigger rain precaution for destination 1.2 mm');
  assert.ok(ids.includes('journey-wind'), 'Should trigger wind precaution for destination 28 km/h');
  assert.ok(analysis.planningExplanation.limitationNotice.includes('Endpoint Weather Notice'));
});

test('8. buildJourneyWeatherPrecautions provides fallback when weather is unavailable', () => {
  const route = {
    startLabel: 'Point A',
    destinationLabel: 'Point B',
    distanceKm: 50,
    durationSeconds: 3600,
    availableRangeKm: 150
  };
  const analysis = buildJourneyWeatherPrecautions(null, null, route);
  assert.equal(analysis.hasAnyWeather, false);
  assert.equal(analysis.precautions[0].id, 'journey-weather-unavailable');
});
