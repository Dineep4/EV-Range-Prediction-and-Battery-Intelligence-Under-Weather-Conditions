import test from "node:test";
import assert from "node:assert/strict";
import {
  buildDrivingRecommendations,
  buildPlanningConsumptionProfile,
  calculateWeatherAdjustedConsumption,
  conservativeWeatherFactor,
  estimatePhysicsRange,
  WEATHER_CONSUMPTION_CONFIG
} from "../src/planning.js";

test("physics range uses kWh, SOC, planning adjustment, Wh/km, then weather factor", () => {
  const result = estimatePhysicsRange({ batteryCapacityKwh: 60, socPercent: 80, planningHealthPercent: 90, consumptionWhKm: 150, weatherFactor: 0.95 });
  assert.equal(result.usableEnergyKwh, 43.2); assert.equal(result.unadjustedRangeKm, 288); assert.ok(Math.abs(result.estimatedRangeKm - 273.6) < 1e-10);
});
test("weather factor is bounded and neutral when temperature is unavailable", () => {
  assert.equal(conservativeWeatherFactor(null), 1); assert.equal(conservativeWeatherFactor(20), 1); assert.ok(conservativeWeatherFactor(-30) >= 0.82); assert.ok(conservativeWeatherFactor(-30) <= 1);
});
test("illustrative planning profile scales with selected consumption", () => {
  const lower=buildPlanningConsumptionProfile(150), higher=buildPlanningConsumptionProfile(200);
  assert.equal(lower.length,8); assert.ok(higher[0].value>lower[0].value); assert.deepEqual(lower.map(x=>x.label),higher.map(x=>x.label)); assert.deepEqual(buildPlanningConsumptionProfile(NaN),[]);
});
test("recommendations explain insufficient range and unavailable charging data", () => {
  const result=buildDrivingRecommendations({socPercent:18,tripDistanceKm:300,estimatedRangeKm:250,stationStatus:"unavailable",stationCount:0});
  assert.deepEqual(result.map(x=>x.id),["range-insufficient","low-soc","stations-unverified"]); assert.match(result[0].trigger,/route distance exceeds/); assert.match(result[2].action,/trusted charging network/);
});
test("consumption recommendation compares only with selected EPA reference", () => {
  const result=buildDrivingRecommendations({socPercent:80,energyConsumptionWhKm:220,epaBaselineWhKm:165}); assert.equal(result[0].id,"high-consumption"); assert.match(result[0].reason,/EPA combined-consumption reference/);
});
test("ambient advice distinguishes live ambient temperature from battery temperature", () => {
  const result=buildDrivingRecommendations({socPercent:80,tripDistanceKm:100,estimatedRangeKm:150,ambientTemperatureC:39});
  assert.equal(result[0].id,"range-margin"); assert.equal(result[1].id,"ambient-weather"); assert.match(result[1].action,/not measured battery temperature/);
});

test("weather-adjusted consumption returns base value when weather is unavailable or disabled", () => {
  const resNull = calculateWeatherAdjustedConsumption(165, null, true);
  assert.equal(resNull.baseConsumptionWhKm, 165);
  assert.equal(resNull.adjustedConsumptionWhKm, 165);
  assert.equal(resNull.isWeatherAdjusted, false);
  assert.equal(resNull.weatherMultiplier, 1.0);

  const resDisabled = calculateWeatherAdjustedConsumption(165, { temperature: 5 }, false);
  assert.equal(resDisabled.adjustedConsumptionWhKm, 165);
  assert.equal(resDisabled.isWeatherAdjusted, false);

  const resMissingTemp = calculateWeatherAdjustedConsumption(165, { temperature: null }, true);
  assert.equal(resMissingTemp.adjustedConsumptionWhKm, 165);
  assert.equal(resMissingTemp.isWeatherAdjusted, false);
});

test("weather-adjusted consumption is neutral in optimal temperature window (18-22 C)", () => {
  const res = calculateWeatherAdjustedConsumption(160, { temperature: 20, wind: 10, precipitation: 0, humidity: 50 }, true);
  assert.equal(res.adjustedConsumptionWhKm, 160);
  assert.equal(res.isWeatherAdjusted, false);
  assert.equal(res.weatherMultiplier, 1.0);
});

test("weather-adjusted consumption increases in cold temperatures", () => {
  // At 0°C, delta from 18°C is 18°C. Penalty = 18 * 0.007 = 0.126 (+12.6%)
  const res = calculateWeatherAdjustedConsumption(100, { temperature: 0, wind: 5, precipitation: 0 }, true);
  assert.ok(res.adjustedConsumptionWhKm > 100);
  assert.equal(res.adjustedConsumptionWhKm, 112.6);
  assert.equal(res.isWeatherAdjusted, true);
  assert.ok(res.reason.includes("0.0°C"));
});

test("weather-adjusted consumption increases in hot temperatures", () => {
  // At 32°C, delta from 22°C is 10°C. Penalty = 10 * 0.005 = 0.05 (+5.0%)
  const res = calculateWeatherAdjustedConsumption(200, { temperature: 32, wind: 10, precipitation: 0 }, true);
  assert.ok(res.adjustedConsumptionWhKm > 200);
  assert.equal(res.adjustedConsumptionWhKm, 210);
  assert.equal(res.isWeatherAdjusted, true);
});

test("weather-adjusted consumption accounts for wind and rain", () => {
  // 35 km/h wind -> (35 - 15) * 0.001 = +0.02
  // 5 mm rain -> 5 * 0.008 = +0.04
  const res = calculateWeatherAdjustedConsumption(150, { temperature: 20, wind: 35, precipitation: 5 }, true);
  assert.ok(res.adjustedConsumptionWhKm > 150);
  assert.ok(res.factors.windFactor > 1.0);
  assert.ok(res.factors.rainFactor > 1.0);
});

test("weather-adjusted consumption multiplier is safely bounded", () => {
  const extremeCold = calculateWeatherAdjustedConsumption(100, { temperature: -50, wind: 100, precipitation: 50 }, true);
  assert.ok(extremeCold.weatherMultiplier <= WEATHER_CONSUMPTION_CONFIG.MAX_WEATHER_MULTIPLIER);
  assert.ok(Number.isFinite(extremeCold.adjustedConsumptionWhKm));
});

test("weather-adjusted consumption handles invalid base consumption safely", () => {
  assert.equal(calculateWeatherAdjustedConsumption(0, { temperature: 25 }, true).adjustedConsumptionWhKm, 0);
  assert.equal(calculateWeatherAdjustedConsumption(-10, { temperature: 25 }, true).adjustedConsumptionWhKm, 0);
  assert.equal(calculateWeatherAdjustedConsumption(NaN, { temperature: 25 }, true).adjustedConsumptionWhKm, 0);
});

test("repeated calculations do not compound adjustment on base consumption", () => {
  const base = 165;
  const weather = { temperature: 5, wind: 20, precipitation: 1, humidity: 60 };
  const run1 = calculateWeatherAdjustedConsumption(base, weather, true);
  const run2 = calculateWeatherAdjustedConsumption(base, weather, true);
  const run3 = calculateWeatherAdjustedConsumption(base, weather, true);
  assert.equal(run1.adjustedConsumptionWhKm, run2.adjustedConsumptionWhKm);
  assert.equal(run2.adjustedConsumptionWhKm, run3.adjustedConsumptionWhKm);
  assert.equal(run1.baseConsumptionWhKm, base);
});

test("range calculation recalculates inversely with weather-adjusted consumption", () => {
  const capacity = 60;
  const soc = 80;
  const soh = 90;
  // Effective battery energy = 60 * 0.8 * 0.9 = 43.2 kWh
  const effectiveEnergyKwh = capacity * (soc / 100) * (soh / 100);

  const baseConsumption = 150;
  const coldWeather = { temperature: 0, wind: 10, precipitation: 0 };
  const mildWeather = { temperature: 20, wind: 10, precipitation: 0 };

  const adjustedCold = calculateWeatherAdjustedConsumption(baseConsumption, coldWeather, true);
  const adjustedMild = calculateWeatherAdjustedConsumption(baseConsumption, mildWeather, true);

  const rangeCold = (effectiveEnergyKwh * 1000) / adjustedCold.adjustedConsumptionWhKm;
  const rangeMild = (effectiveEnergyKwh * 1000) / adjustedMild.adjustedConsumptionWhKm;

  // Higher consumption in cold weather produces lower driving range
  assert.ok(adjustedCold.adjustedConsumptionWhKm > adjustedMild.adjustedConsumptionWhKm);
  assert.ok(rangeCold < rangeMild);

  // Changing SOC immediately scales range proportionally
  const rangeColdHalfSoc = (capacity * (40 / 100) * (soh / 100) * 1000) / adjustedCold.adjustedConsumptionWhKm;
  assert.ok(Math.abs(rangeCold - rangeColdHalfSoc * 2) < 1e-6);
});

test("simulation mode tests: 32C, 35C, and 37C produce distinct, sensitive consumption and range updates", () => {
  const capacity = 60;
  const soc = 82;
  const soh = 92;
  const baseConsumption = 165;
  // Effective battery energy = 60 * 0.82 * 0.92 = 45.264 kWh
  const effectiveEnergyKwh = capacity * (soc / 100) * (soh / 100);

  // Live API weather baseline (e.g. 26.5°C)
  const liveWeather = { temperature: 26.5, wind: 10, precipitation: 0, humidity: 55 };
  const resLive = calculateWeatherAdjustedConsumption(baseConsumption, liveWeather, true);
  const rangeLive = (effectiveEnergyKwh * 1000) / resLive.adjustedConsumptionWhKm;

  // Simulation mode at 32°C, 35°C, 37°C
  const sim32 = calculateWeatherAdjustedConsumption(baseConsumption, { ...liveWeather, temperature: 32 }, true);
  const sim35 = calculateWeatherAdjustedConsumption(baseConsumption, { ...liveWeather, temperature: 35 }, true);
  const sim37 = calculateWeatherAdjustedConsumption(baseConsumption, { ...liveWeather, temperature: 37 }, true);

  const range32 = (effectiveEnergyKwh * 1000) / sim32.adjustedConsumptionWhKm;
  const range35 = (effectiveEnergyKwh * 1000) / sim35.adjustedConsumptionWhKm;
  const range37 = (effectiveEnergyKwh * 1000) / sim37.adjustedConsumptionWhKm;

  // 1. Consumption increases monotonically from 32°C to 35°C to 37°C
  assert.ok(sim32.adjustedConsumptionWhKm < sim35.adjustedConsumptionWhKm, "35°C must have higher consumption than 32°C");
  assert.ok(sim35.adjustedConsumptionWhKm < sim37.adjustedConsumptionWhKm, "37°C must have higher consumption than 35°C");

  // 2. Range decreases monotonically from 32°C to 35°C to 37°C
  assert.ok(range32 > range35, "35°C must have lower range than 32°C");
  assert.ok(range35 > range37, "37°C must have lower range than 35°C");

  // 3. Reset to live weather restores exact live baseline
  const resetLive = calculateWeatherAdjustedConsumption(baseConsumption, liveWeather, true);
  const rangeReset = (effectiveEnergyKwh * 1000) / resetLive.adjustedConsumptionWhKm;
  assert.equal(resetLive.adjustedConsumptionWhKm, resLive.adjustedConsumptionWhKm);
  assert.equal(rangeReset, rangeLive);
});


