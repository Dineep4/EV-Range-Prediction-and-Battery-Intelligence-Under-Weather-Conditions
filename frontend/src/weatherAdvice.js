/**
 * VoltWise — Context-Aware Weather Advice Logic
 *
 * Implements rule-based, transparent weather-informed recommendations for:
 * 1. Dashboard: Local Weather & EV Care (parking, preconditioning, battery care)
 * 2. Trip Planner: Journey Weather & Route Precautions (endpoint observations, road conditions, energy model explanation)
 *
 * NOTE: These are advisory guidelines based on physics heuristics and driving best practices.
 * Ambient temperature is an external atmospheric measurement, not measured internal battery temperature.
 * Weather heuristics are transparent planning aids, not validated physical degradation or ML models.
 */

export const DASHBOARD_WEATHER_THRESHOLDS = {
  HOT_TEMP_C: 30,          // Ambient temp >= 30°C: high-heat parking & cooling advice
  COLD_TEMP_C: 10,         // Ambient temp <= 10°C: cold-range buffer & preconditioning advice
  RAIN_PRECIPITATION_MM: 0.1, // Precipitation >= 0.1 mm: wet road precautions
  STRONG_WIND_KMH: 25      // Wind >= 25 km/h: crosswind & aerodynamic drag caution
};

/**
 * Builds compact, relevant EV care recommendations for the Dashboard Local Weather card.
 *
 * @param {Object|null} weather - Live weather object with temperature, humidity, wind, precipitation, time
 * @param {string} weatherState - 'live' | 'stale' | 'loading' | 'unavailable'
 * @returns {Object} { status, freshnessLabel, observedTime, items, disclaimer }
 */
export function buildDashboardWeatherEvCareAdvice(weather, weatherState = 'unavailable') {
  const isAvailable = weatherState === 'live' || weatherState === 'stale';
  const hasValidTemp = weather && Number.isFinite(Number(weather.temperature));

  if (!isAvailable || !hasValidTemp) {
    return {
      status: weatherState === 'loading' ? 'loading' : 'unavailable',
      freshnessLabel: weatherState === 'loading' ? 'Loading weather...' : 'Weather unavailable',
      observedTime: null,
      items: [
        {
          id: 'care-unavailable',
          title: 'Live weather advice unavailable',
          text: 'Current weather observations could not be retrieved. Follow standard EV operating guidance and vehicle dashboard alerts.',
          severity: 'neutral'
        }
      ],
      disclaimer: 'Ambient atmospheric temperature is not internal battery temperature. Consult vehicle instrumentation.'
    };
  }

  const temp = Number(weather.temperature);
  const wind = Number.isFinite(Number(weather.wind)) ? Number(weather.wind) : 0;
  const precip = Number.isFinite(Number(weather.precipitation)) ? Number(weather.precipitation) : 0;

  const isHot = temp >= DASHBOARD_WEATHER_THRESHOLDS.HOT_TEMP_C;
  const isCold = temp <= DASHBOARD_WEATHER_THRESHOLDS.COLD_TEMP_C;
  const isRain = precip >= DASHBOARD_WEATHER_THRESHOLDS.RAIN_PRECIPITATION_MM;
  const isWindy = wind >= DASHBOARD_WEATHER_THRESHOLDS.STRONG_WIND_KMH;
  const isMild = !isHot && !isCold && !isRain && !isWindy;

  const items = [];

  if (isHot) {
    items.push({
      id: 'care-hot-parking',
      title: 'Shaded or covered parking',
      text: 'Prefer shaded or covered parking where practical to limit solar heat absorption into the cabin and battery pack. Follow vehicle temperature warnings and manufacturer thermal guidance.',
      severity: 'warning'
    });
  }

  if (isCold) {
    items.push({
      id: 'care-cold-range',
      title: 'Cold range & preconditioning',
      text: 'Expect potentially reduced range due to lower cell reaction rates and heating demand. Consider preconditioning the cabin and battery while plugged in if supported by your vehicle.',
      severity: 'warning'
    });
  }

  if (isRain) {
    items.push({
      id: 'care-rain-safety',
      title: 'Wet-road safety & visibility',
      text: 'Prioritise wet-road safety: reduce cruising speed, increase following distance, and avoid flooded roads or standing water.',
      severity: 'warning'
    });
  }

  if (isWindy) {
    items.push({
      id: 'care-wind-crosswind',
      title: 'Crosswind & aerodynamic caution',
      text: 'Strong winds increase aerodynamic drag and may buffet high-profile vehicles on exposed roads. Maintain steady lane positioning.',
      severity: 'info'
    });
  }

  if (isMild) {
    items.push({
      id: 'care-mild-optimal',
      title: 'Mild operating conditions',
      text: 'Current ambient conditions are within standard operational bounds. No special weather-related precaution is currently triggered by the available data.',
      severity: 'positive'
    });
  }

  const freshnessLabel = weatherState === 'live'
    ? 'Live observation'
    : 'Cached observation';

  return {
    status: weatherState,
    freshnessLabel,
    observedTime: weather.time || null,
    items,
    disclaimer: 'Ambient temperature is an external atmospheric reading, not internal battery pack temperature. Ambient heat does not prove pack overheating, and weather heuristics are informative planning aids, not validated physical degradation models.'
  };
}

/**
 * Builds journey-specific weather precautions for Trip Planner.
 *
 * @param {Object|null} departureWeather - Weather at route start
 * @param {Object|null} destinationWeather - Weather at route destination
 * @param {Object} route - Calculated route object
 * @returns {Object} Detailed journey weather analysis & precautions
 */
export function buildJourneyWeatherPrecautions(departureWeather, destinationWeather, route) {
  if (!route) return null;

  const depValid = Boolean(departureWeather && Number.isFinite(Number(departureWeather.temperature)));
  const destValid = Boolean(destinationWeather && Number.isFinite(Number(destinationWeather.temperature)));

  const hasAnyWeather = depValid || destValid;

  const depTemp = depValid ? Number(departureWeather.temperature) : null;
  const destTemp = destValid ? Number(destinationWeather.temperature) : null;

  const maxTemp = Math.max(...[depTemp, destTemp].filter(t => t !== null));
  const minTemp = Math.min(...[depTemp, destTemp].filter(t => t !== null));

  const maxPrecip = Math.max(...[
    depValid ? Number(departureWeather.precipitation || 0) : 0,
    destValid ? Number(destinationWeather.precipitation || 0) : 0
  ]);

  const maxWind = Math.max(...[
    depValid ? Number(departureWeather.wind || 0) : 0,
    destValid ? Number(destinationWeather.wind || 0) : 0
  ]);

  const precautions = [];

  if (!hasAnyWeather) {
    precautions.push({
      id: 'journey-weather-unavailable',
      title: 'Journey endpoint weather unavailable',
      description: 'Live endpoint weather could not be retrieved for this route. Route planning physics remains active. Observe local road conditions and consult regional weather forecasts.',
      severity: 'neutral'
    });
  } else {
    // Rain condition
    if (maxPrecip >= DASHBOARD_WEATHER_THRESHOLDS.RAIN_PRECIPITATION_MM) {
      precautions.push({
        id: 'journey-rain',
        title: 'Wet-road caution along journey',
        description: `Precipitation observed at journey endpoints (${maxPrecip.toFixed(1)} mm). Water displacement increases tyre rolling resistance and braking distance. Reduce cruising speed, increase following distance, and exercise extra caution on highways.`,
        severity: 'warning'
      });
    }

    // Hot condition
    if (maxTemp >= DASHBOARD_WEATHER_THRESHOLDS.HOT_TEMP_C) {
      precautions.push({
        id: 'journey-heat',
        title: 'High ambient temperature caution',
        description: `Elevated ambient temperature observed (${maxTemp.toFixed(1)}°C; threshold: ≥${DASHBOARD_WEATHER_THRESHOLDS.HOT_TEMP_C}°C). Check vehicle status before departure, plan shaded rest stops for extended trips, avoid prolonged parking in direct sunlight where practical, and follow vehicle thermal alerts.`,
        severity: 'warning'
      });
    }

    // Cold condition
    if (minTemp <= DASHBOARD_WEATHER_THRESHOLDS.COLD_TEMP_C) {
      precautions.push({
        id: 'journey-cold',
        title: 'Cold-weather journey range buffer',
        description: `Cold ambient temperatures observed (${minTemp.toFixed(1)}°C; threshold: ≤${DASHBOARD_WEATHER_THRESHOLDS.COLD_TEMP_C}°C). Electrochemical efficiency drops and cabin heating increases consumption. Maintain an additional 15–20% range reserve over planned distance and check vehicle-specific cold-weather guidance.`,
        severity: 'warning'
      });
    }

    // Wind condition
    if (maxWind >= DASHBOARD_WEATHER_THRESHOLDS.STRONG_WIND_KMH) {
      precautions.push({
        id: 'journey-wind',
        title: 'Strong winds & bridge caution',
        description: `Elevated wind speed reported (${maxWind.toFixed(0)} km/h; threshold: ≥${DASHBOARD_WEATHER_THRESHOLDS.STRONG_WIND_KMH} km/h). Aerodynamic drag increases significantly at highway speeds; exercise caution on exposed highways, overpasses, and bridges.`,
        severity: 'info'
      });
    }

    // Neutral condition
    if (precautions.length === 0) {
      precautions.push({
        id: 'journey-neutral',
        title: 'Favorable travel weather',
        description: `No significant adverse weather concern detected at journey endpoints (${minTemp.toFixed(1)}°C to ${maxTemp.toFixed(1)}°C, mild winds, dry). This is an informational weather advisory, not a guarantee of safe travel or road clearance.`,
        severity: 'positive'
      });
    }
  }

  return {
    hasAnyWeather,
    departureWeather: depValid ? departureWeather : null,
    destinationWeather: destValid ? destinationWeather : null,
    precautions,
    planningExplanation: {
      physicsFormula: 'Usable Energy (kWh) = Capacity × (SOC / 100) × (SOH / 100); Nominal Range = Usable Energy × 1000 / Consumption Wh/km.',
      weatherAdjustment: 'The optional weather adjustment applies a conservative heuristic (1 - 0.003 per °C outside 15–25°C, bounded 0.82–1.00). It is a transparent planning aid, not an empirical wind-tunnel or validated climate model.',
      limitationNotice: 'Endpoint Weather Notice: Open-Meteo current conditions endpoint provides localized endpoint observations at departure and destination. Continuous microclimate forecasts along the entire road geometry are not provided by this endpoint. These readings are local endpoint observations, not multi-hour future forecasts.'
    }
  };
}
