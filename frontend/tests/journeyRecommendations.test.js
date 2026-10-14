import test from 'node:test';
import assert from 'node:assert/strict';
import {
  WEATHER_THRESHOLDS,
  classifyWeatherCondition,
  buildWeatherPrecautions,
  buildSpeedAndDrivingGuidance,
  buildBatteryAndLoadPrecautions,
  buildParkingRecommendations,
  DEFAULT_CHECKLIST_ITEMS
} from '../src/journeyRecommendations.js';

test('WEATHER_THRESHOLDS constants have expected sensible defaults', () => {
  assert.equal(WEATHER_THRESHOLDS.HOT_TEMP_C, 30);
  assert.equal(WEATHER_THRESHOLDS.COLD_TEMP_C, 10);
  assert.equal(WEATHER_THRESHOLDS.STRONG_WIND_KMH, 25);
  assert.equal(WEATHER_THRESHOLDS.RAIN_PRECIPITATION_MM, 0.1);
});

test('classifyWeatherCondition handles missing or incomplete weather data gracefully', () => {
  const missing1 = classifyWeatherCondition(null);
  assert.equal(missing1.isAvailable, false);
  assert.equal(missing1.temperatureC, null);

  const missing2 = classifyWeatherCondition({});
  assert.equal(missing2.isAvailable, false);

  const missing3 = classifyWeatherCondition({ temperature: 'invalid' });
  assert.equal(missing3.isAvailable, false);

  const precautions = buildWeatherPrecautions(null);
  assert.equal(precautions.length, 1);
  assert.equal(precautions[0].id, 'weather-unavailable');
  assert.match(precautions[0].detail, /Weather data unavailable/i);
});

test('buildWeatherPrecautions generates appropriate precautions for hot weather (>=30°C)', () => {
  const hotWeather = {
    temperature: 36,
    humidity: 45,
    wind: 12,
    precipitation: 0
  };
  const classification = classifyWeatherCondition(hotWeather);
  assert.equal(classification.isAvailable, true);
  assert.equal(classification.isHot, true);
  assert.equal(classification.isCold, false);

  const precautions = buildWeatherPrecautions(hotWeather);
  const ids = precautions.map((p) => p.id);

  assert.ok(ids.includes('hot-speed'), 'Should caution against unnecessary high-speed driving');
  assert.ok(ids.includes('hot-cooling'), 'Should recommend sensible cabin pre-cooling');
  assert.ok(ids.includes('hot-parking'), 'Should recommend shaded parking');
  assert.ok(ids.includes('hot-safety-critical'), 'Should include critical warning about children/pets');
  assert.ok(ids.includes('hot-alerts'), 'Should advise checking vehicle warnings');
});

test('buildWeatherPrecautions generates appropriate precautions for cold weather (<=10°C)', () => {
  const coldWeather = {
    temperature: 4,
    humidity: 70,
    wind: 10,
    precipitation: 0
  };
  const classification = classifyWeatherCondition(coldWeather);
  assert.equal(classification.isAvailable, true);
  assert.equal(classification.isCold, true);
  assert.equal(classification.isHot, false);

  const precautions = buildWeatherPrecautions(coldWeather);
  const ids = precautions.map((p) => p.id);

  assert.ok(ids.includes('cold-efficiency'), 'Should explain reduced battery efficiency');
  assert.ok(ids.includes('cold-precondition'), 'Should suggest preconditioning while plugged in');
  assert.ok(ids.includes('cold-reserve'), 'Should suggest planning a range buffer');
  assert.ok(ids.includes('cold-driving'), 'Should suggest smooth driving');
});

test('buildWeatherPrecautions detects rain and standing water precautions when precipitation is recorded', () => {
  const rainyWeather = {
    temperature: 22,
    humidity: 88,
    wind: 15,
    precipitation: 2.4
  };
  const classification = classifyWeatherCondition(rainyWeather);
  assert.equal(classification.isRain, true);

  const precautions = buildWeatherPrecautions(rainyWeather);
  const ids = precautions.map((p) => p.id);

  assert.ok(ids.includes('rain-safety'), 'Should recommend increased following distance and lower speed');
  assert.ok(ids.includes('rain-standing-water'), 'Should advise about standing water and rolling resistance');
});

test('buildWeatherPrecautions detects strong wind and aerodynamic precautions (>=25 km/h)', () => {
  const windyWeather = {
    temperature: 20,
    humidity: 50,
    wind: 32,
    precipitation: 0
  };
  const classification = classifyWeatherCondition(windyWeather);
  assert.equal(classification.isWindy, true);

  const precautions = buildWeatherPrecautions(windyWeather);
  const ids = precautions.map((p) => p.id);

  assert.ok(ids.includes('wind-aerodynamics'), 'Should explain headwinds and aerodynamic drag');
});

test('buildWeatherPrecautions provides positive guidance for moderate conditions', () => {
  const moderateWeather = {
    temperature: 22,
    humidity: 55,
    wind: 12,
    precipitation: 0
  };
  const classification = classifyWeatherCondition(moderateWeather);
  assert.equal(classification.isModerate, true);

  const precautions = buildWeatherPrecautions(moderateWeather);
  const ids = precautions.map((p) => p.id);

  assert.ok(ids.includes('moderate-ideal'), 'Should note favorable operating conditions');
  assert.equal(precautions[0].severity, 'positive');
});

test('buildSpeedAndDrivingGuidance emphasizes safety over range and avoids synthetic speed prescription', () => {
  const route = { distanceKm: 85, durationSeconds: 3600 };
  const hotWeather = { temperature: 35, wind: 10, precipitation: 0 };

  const guidance = buildSpeedAndDrivingGuidance(hotWeather, route);
  assert.ok(guidance.length >= 3);

  const fullText = guidance.map((g) => `${g.title} ${g.text}`).join(' ');
  assert.match(fullText, /posted speed limits/i);
  assert.match(fullText, /smooth acceleration/i);
  assert.match(fullText, /does NOT prescribe an artificial/i);
});

test('buildBatteryAndLoadPrecautions covers cargo, reserve, tyre pressure, and disclaims live measurement', () => {
  const precautions = buildBatteryAndLoadPrecautions(80, 60, 165);
  assert.ok(precautions.length >= 4);

  const fullText = precautions.map((p) => `${p.title} ${p.text}`).join(' ');
  assert.match(fullText, /cargo/i);
  assert.match(fullText, /reserve/i);
  assert.match(fullText, /tyre pressure/i);
  assert.match(fullText, /does NOT measure actual real-time/i);
});

test('buildParkingRecommendations distinguishes hot, cold, and wet scenarios appropriately', () => {
  const hotParking = buildParkingRecommendations({ temperature: 34 });
  assert.match(hotParking[0].title, /shaded or covered/i);

  const coldParking = buildParkingRecommendations({ temperature: 2 });
  assert.match(coldParking[0].title, /freezing/i);

  const rainParking = buildParkingRecommendations({ temperature: 20, precipitation: 5.0 });
  const hasFloodNotice = rainParking.some((p) => /flood/i.test(p.title) || /flood/i.test(p.text));
  assert.ok(hasFloodNotice, 'Should caution against flood-prone parking areas in rain');
});

test('DEFAULT_CHECKLIST_ITEMS provides 5 pre-departure verification items', () => {
  assert.equal(DEFAULT_CHECKLIST_ITEMS.length, 5);
  const ids = DEFAULT_CHECKLIST_ITEMS.map((item) => item.id);
  assert.deepEqual(ids, [
    'charge_confirmed',
    'tyres_checked',
    'cable_packed',
    'weather_reviewed',
    'charging_identified'
  ]);
});
