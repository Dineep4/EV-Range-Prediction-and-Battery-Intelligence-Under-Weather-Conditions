export const WEATHER_ADJUSTMENT_DESCRIPTION = "Planning heuristic only: 1 - 0.003 per degree outside 15 to 25 C, bounded from 0.82 to 1.00. It is not a weather-trained model.";

/**
 * Configurable rule-based coefficients for weather-dependent energy consumption.
 * Based on empirical electric vehicle energy studies (HVAC thermal loads, aerodynamic drag, rolling resistance).
 * NOTE: These adjustments are rule-based physics heuristics for planning estimation, not ML model predictions.
 */
export const WEATHER_CONSUMPTION_CONFIG = {
  OPTIMAL_TEMP_MIN_C: 18,
  OPTIMAL_TEMP_MAX_C: 22,
  COLD_PENALTY_PER_C: 0.007, // +0.7% Wh/km per °C below 18°C
  MAX_COLD_PENALTY: 0.30,    // Cap at +30%
  HOT_PENALTY_PER_C: 0.005,  // +0.5% Wh/km per °C above 22°C
  MAX_HOT_PENALTY: 0.20,     // Cap at +20%
  WIND_CALM_THRESHOLD_KMH: 15,
  WIND_PENALTY_PER_KMH: 0.001, // +0.1% Wh/km per km/h above 15 km/h
  MAX_WIND_PENALTY: 0.08,      // Cap at +8%
  RAIN_THRESHOLD_MM: 0.1,
  RAIN_PENALTY_PER_MM: 0.008,  // +0.8% Wh/km per mm precipitation
  MAX_RAIN_PENALTY: 0.06,      // Cap at +6%
  MIN_WEATHER_MULTIPLIER: 0.85,
  MAX_WEATHER_MULTIPLIER: 1.50,
};

export function calculateWeatherAdjustedConsumption(baseConsumptionWhKm, weather, isEnabled = true) {
  const base = Number(baseConsumptionWhKm);
  if (!Number.isFinite(base) || base <= 0) {
    return {
      baseConsumptionWhKm: 0,
      adjustedConsumptionWhKm: 0,
      weatherMultiplier: 1.0,
      isWeatherAdjusted: false,
      factors: { tempFactor: 1, windFactor: 1, rainFactor: 1, humidityFactor: 1 },
      reason: "Invalid base consumption"
    };
  }

  if (!isEnabled || !weather || weather.temperature === null || weather.temperature === undefined || !Number.isFinite(Number(weather.temperature))) {
    return {
      baseConsumptionWhKm: base,
      adjustedConsumptionWhKm: base,
      weatherMultiplier: 1.0,
      isWeatherAdjusted: false,
      factors: { tempFactor: 1, windFactor: 1, rainFactor: 1, humidityFactor: 1 },
      reason: isEnabled ? "Live weather observation unavailable" : "Weather adjustment disabled"
    };
  }

  const temp = Number(weather.temperature);
  const wind = Number(weather.wind);
  const rain = Number(weather.precipitation);
  const humidity = Number(weather.humidity);

  // 1. Temperature factor (primary)
  let tempDelta = 0;
  if (temp < WEATHER_CONSUMPTION_CONFIG.OPTIMAL_TEMP_MIN_C) {
    tempDelta = Math.min(
      WEATHER_CONSUMPTION_CONFIG.MAX_COLD_PENALTY,
      (WEATHER_CONSUMPTION_CONFIG.OPTIMAL_TEMP_MIN_C - temp) * WEATHER_CONSUMPTION_CONFIG.COLD_PENALTY_PER_C
    );
  } else if (temp > WEATHER_CONSUMPTION_CONFIG.OPTIMAL_TEMP_MAX_C) {
    tempDelta = Math.min(
      WEATHER_CONSUMPTION_CONFIG.MAX_HOT_PENALTY,
      (temp - WEATHER_CONSUMPTION_CONFIG.OPTIMAL_TEMP_MAX_C) * WEATHER_CONSUMPTION_CONFIG.HOT_PENALTY_PER_C
    );
  }
  const tempFactor = 1 + tempDelta;

  // 2. Wind factor (aerodynamic drag)
  let windDelta = 0;
  if (Number.isFinite(wind) && wind > WEATHER_CONSUMPTION_CONFIG.WIND_CALM_THRESHOLD_KMH) {
    windDelta = Math.min(
      WEATHER_CONSUMPTION_CONFIG.MAX_WIND_PENALTY,
      (wind - WEATHER_CONSUMPTION_CONFIG.WIND_CALM_THRESHOLD_KMH) * WEATHER_CONSUMPTION_CONFIG.WIND_PENALTY_PER_KMH
    );
  }
  const windFactor = 1 + windDelta;

  // 3. Precipitation factor (road rolling resistance)
  let rainDelta = 0;
  if (Number.isFinite(rain) && rain > WEATHER_CONSUMPTION_CONFIG.RAIN_THRESHOLD_MM) {
    rainDelta = Math.min(
      WEATHER_CONSUMPTION_CONFIG.MAX_RAIN_PENALTY,
      rain * WEATHER_CONSUMPTION_CONFIG.RAIN_PENALTY_PER_MM
    );
  }
  const rainFactor = 1 + rainDelta;

  // 4. Relative humidity (HVAC latent dehumidifier load in non-mild conditions)
  let humidityDelta = 0;
  if (Number.isFinite(humidity) && humidity > 80 && (temp > 28 || temp < 10)) {
    humidityDelta = 0.015;
  }
  const humidityFactor = 1 + humidityDelta;

  const rawMultiplier = tempFactor * windFactor * rainFactor * humidityFactor;
  const weatherMultiplier = Math.max(
    WEATHER_CONSUMPTION_CONFIG.MIN_WEATHER_MULTIPLIER,
    Math.min(WEATHER_CONSUMPTION_CONFIG.MAX_WEATHER_MULTIPLIER, rawMultiplier)
  );

  const adjustedConsumptionWhKm = Math.round(base * weatherMultiplier * 10) / 10;

  return {
    baseConsumptionWhKm: base,
    adjustedConsumptionWhKm,
    weatherMultiplier,
    isWeatherAdjusted: weatherMultiplier !== 1.0,
    factors: { tempFactor, windFactor, rainFactor, humidityFactor },
    reason: weatherMultiplier !== 1.0
      ? `Adjusted for live weather (${temp.toFixed(1)}°C${wind > 15 ? ` · ${wind.toFixed(0)} km/h wind` : ""}${rain > 0.1 ? ` · ${rain.toFixed(1)} mm rain` : ""})`
      : "Live conditions within optimal vehicle efficiency window (no penalty)"
  };
}

export function conservativeWeatherFactor(temperatureC) {
  if (temperatureC === null || temperatureC === undefined || temperatureC === "" || !Number.isFinite(Number(temperatureC))) return 1;
  return Math.max(0.82, Math.min(1, 1 - Math.max(0, Math.abs(Number(temperatureC) - 20) - 5) * 0.003));
}
export function estimatePhysicsRange({ batteryCapacityKwh, socPercent, planningHealthPercent, consumptionWhKm, weatherFactor = 1 }) {
  const inputs = [batteryCapacityKwh, socPercent, planningHealthPercent, consumptionWhKm, weatherFactor].map(Number);
  if (inputs.some(value => !Number.isFinite(value))) throw new Error("Range inputs must be finite numbers.");
  const [capacity, soc, health, consumption, weather] = inputs;
  if (capacity <= 0 || soc < 0 || soc > 100 || health <= 0 || health > 120 || consumption <= 0 || weather < 0.5 || weather > 1.2) throw new Error("Range inputs are outside the supported planning limits.");
  const usableEnergyKwh = capacity * (soc / 100) * (health / 100);
  const unadjustedRangeKm = usableEnergyKwh * 1000 / consumption;
  return { usableEnergyKwh, unadjustedRangeKm, estimatedRangeKm: unadjustedRangeKm * weather, weatherFactor: weather };
}
const PROFILE_MULTIPLIERS = [0.88, 0.94, 0.92, 1.03, 1.0, 0.96, 1.12, 1.01];
const PROFILE_LABELS = ["06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00"];
export function buildPlanningConsumptionProfile(averageWhKm) {
  const average = Number(averageWhKm);
  if (!Number.isFinite(average) || average <= 0) return [];
  return PROFILE_LABELS.map((label, index) => ({ label, value: Math.round(average * PROFILE_MULTIPLIERS[index]) }));
}
export function buildDrivingRecommendations({ socPercent, tripDistanceKm, estimatedRangeKm, stationStatus, stationCount = 0, ambientTemperatureC, energyConsumptionWhKm, epaBaselineWhKm }) {
  const recommendations = [];
  const soc = Number(socPercent), distance = Number(tripDistanceKm), range = Number(estimatedRangeKm), consumption = Number(energyConsumptionWhKm), baseline = Number(epaBaselineWhKm);
  if (Number.isFinite(distance) && Number.isFinite(range)) {
    const remaining = range - distance;
    if (remaining < 0) recommendations.push({ id: "range-insufficient", priority: "High", title: "Review the trip range", reason: `The current planning estimate is ${Math.abs(remaining).toFixed(1)} km below the ${distance.toFixed(1)} km route.`, action: stationCount > 0 ? "Review mapped station candidates and verify one with its operator before departure." : "Shorten the trip or identify a charging stop through a trusted charging network before departure.", trigger: "Calculated route distance exceeds calculated available range." });
    else if (range >= distance * 1.25) recommendations.push({ id: "range-margin", priority: "Note", title: "Planning estimate has a margin", reason: `The estimated range is ${remaining.toFixed(1)} km above this route under the current inputs.`, action: "Keep a practical reserve; traffic, terrain, speed, cabin climate and battery condition can change actual range.", trigger: "Calculated range is at least 25% above route distance; this is an app planning prompt, not a guarantee." });
    else recommendations.push({ id: "range-close", priority: "Consider", title: "Range margin is limited", reason: `The estimated range is only ${remaining.toFixed(1)} km above the route.`, action: "Check route conditions and verify a charging option along the way.", trigger: "Calculated range is above route distance but below the app's 25% planning margin." });
  }
  if (Number.isFinite(soc) && soc <= 20) recommendations.push({ id: "low-soc", priority: "Consider", title: "State of charge is low", reason: `The entered state of charge is ${soc.toFixed(0)}%.`, action: "Consider charging before a longer trip and check the vehicle's own range display.", trigger: "User-entered SOC is at or below the app's 20% planning prompt; this is not a universal battery limit." });
  if (Number.isFinite(consumption) && Number.isFinite(baseline) && baseline > 0 && consumption > baseline * 1.2) recommendations.push({ id: "high-consumption", priority: "Note", title: "Consumption input is above the EPA reference", reason: `${consumption.toFixed(0)} Wh/km is over 20% higher than the selected EPA combined-consumption reference (${baseline.toFixed(0)} Wh/km).`, action: "Check whether the entered value reflects current speed, traffic, tyre pressure and climate-control use.", trigger: "User-entered consumption exceeds the selected configuration's converted EPA combined value by more than 20%." });
  const ambient = Number(ambientTemperatureC);
  if (Number.isFinite(ambient) && (ambient < 5 || ambient > 35)) recommendations.push({ id: "ambient-weather", priority: "Note", title: "Ambient conditions may affect range", reason: `The live provider reported ${ambient.toFixed(1)} C ambient temperature.`, action: "Actual effects depend on the vehicle and trip. This is ambient weather, not measured battery temperature; consult vehicle guidance.", trigger: "Live ambient reading is outside the app's 5 to 35 C information prompt." });
  if (["unavailable", "error", "empty"].includes(stationStatus)) recommendations.push({ id: "stations-unverified", priority: "Consider", title: "Charging stops are not verified", reason: stationStatus === "empty" ? "The latest mapped search returned no stations in the selected search area." : "The mapped charging-station provider did not return usable results.", action: "Check a trusted charging network and confirm the station is operating before relying on it.", trigger: "Current charging-station search status." });
  if (!recommendations.length) recommendations.push({ id: "general", priority: "Note", title: "Use the estimate as a planning aid", reason: "No specific planning prompt was triggered by the available inputs.", action: "Follow the vehicle's own warnings and verify current route and charging conditions.", trigger: "No route, low-SOC, high-consumption, unusual-ambient or station-availability rule matched." });
  return recommendations;
}
