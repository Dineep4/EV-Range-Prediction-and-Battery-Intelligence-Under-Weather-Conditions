/**
 * VoltWise — Trip Summary Calculation and Evaluation Logic
 *
 * Implements physics-informed trip metrics, linear arrival SOC estimation,
 * endpoint weather precautions, and charging-stop suggestions.
 */
import { DASHBOARD_WEATHER_THRESHOLDS } from './weatherAdvice.js';

export function calculateDistanceReserve(availableRangeKm, distanceKm) {
  const r = Number(availableRangeKm);
  const d = Number(distanceKm);
  if (!Number.isFinite(r) || !Number.isFinite(d)) {
    return {
      availableRangeKm: 0,
      distanceKm: 0,
      distanceReserveKm: 0,
      shortfallKm: 0,
      isWithinRange: false,
    };
  }
  const distanceReserveKm = r - d;
  const isWithinRange = distanceReserveKm >= 0;
  return {
    availableRangeKm: r,
    distanceKm: d,
    distanceReserveKm,
    shortfallKm: isWithinRange ? 0 : Math.abs(distanceReserveKm),
    isWithinRange,
  };
}

export function calculateArrivalSocPercent(soc, distanceKm, availableRangeKm) {
  const s = Number(soc);
  const d = Number(distanceKm);
  const r = Number(availableRangeKm);
  if (!Number.isFinite(s) || !Number.isFinite(d) || !Number.isFinite(r) || r <= 0 || d < 0 || s <= 0) {
    return 0;
  }
  if (d > r) {
    return 0; // Depleted before arrival
  }
  const fraction = Math.max(0, 1 - d / r);
  return Math.max(0, Math.min(100, Math.round(s * fraction)));
}

export function buildTripWeatherPrecautions(departureWeather, destinationWeather, weatherState) {
  const depW = departureWeather;
  const destW = destinationWeather;
  const hasWeather = Boolean(
    (depW && Number.isFinite(Number(depW.temperature))) ||
    (destW && Number.isFinite(Number(destW.temperature)))
  );

  const tempToCheck = depW && Number.isFinite(Number(depW.temperature)) ? Number(depW.temperature) : null;
  const rainToCheck = depW && Number.isFinite(Number(depW.precipitation)) ? Number(depW.precipitation) : 0;
  const windToCheck = depW && Number.isFinite(Number(depW.wind)) ? Number(depW.wind) : 0;

  if (!hasWeather || weatherState === 'unavailable') {
    return {
      hasWeather: false,
      headline: 'Weather observation unavailable',
      detail: 'Live weather readings could not be retrieved for journey endpoints. Standard physics planning applies.',
      severity: 'neutral',
      disclaimer: 'Weather service unavailable.'
    };
  }

  if (rainToCheck >= DASHBOARD_WEATHER_THRESHOLDS.RAIN_PRECIPITATION_MM) {
    return {
      hasWeather: true,
      headline: `Wet asphalt detected (${rainToCheck.toFixed(1)} mm precipitation)`,
      detail: 'Reduced road grip increases braking distance and rolling resistance. Reduce cruising speed and maintain a larger gap.',
      severity: 'warning',
      disclaimer: 'Local Open-Meteo endpoint observation. Intermediate microclimate forecasts are not available.'
    };
  }

  if (tempToCheck !== null && tempToCheck >= DASHBOARD_WEATHER_THRESHOLDS.HOT_TEMP_C) {
    return {
      hasWeather: true,
      headline: `High ambient temperature (${tempToCheck.toFixed(1)}°C)`,
      detail: 'Pre-cool cabin while connected to charger before departure. Observe vehicle temperature alerts and avoid direct sunlight at stops.',
      severity: 'warning',
      disclaimer: 'Local Open-Meteo endpoint observation. Intermediate microclimate forecasts are not available.'
    };
  }

  if (tempToCheck !== null && tempToCheck <= DASHBOARD_WEATHER_THRESHOLDS.COLD_TEMP_C) {
    return {
      hasWeather: true,
      headline: `Cold ambient conditions (${tempToCheck.toFixed(1)}°C)`,
      detail: 'Cold reduces electrochemical reaction rates and increases heating draw. Maintain an additional 15–20% range reserve.',
      severity: 'warning',
      disclaimer: 'Local Open-Meteo endpoint observation. Intermediate microclimate forecasts are not available.'
    };
  }

  if (windToCheck >= DASHBOARD_WEATHER_THRESHOLDS.STRONG_WIND_KMH) {
    return {
      hasWeather: true,
      headline: `Elevated winds (${windToCheck.toFixed(0)} km/h)`,
      detail: 'Aerodynamic drag increases on open highways and elevated viaducts. Allow additional energy margin.',
      severity: 'info',
      disclaimer: 'Local Open-Meteo endpoint observation. Intermediate microclimate forecasts are not available.'
    };
  }

  return {
    hasWeather: true,
    headline: tempToCheck !== null ? `Mild conditions (${tempToCheck.toFixed(1)}°C)` : 'Mild conditions',
    detail: 'No adverse weather concerns detected at journey endpoints. This is an informational advisory, not a guaranteed forecast.',
    severity: 'positive',
    disclaimer: 'Local Open-Meteo endpoint observation. Intermediate microclimate forecasts are not available.'
  };
}

export function buildTripChargingAdvice({ isWithinRange, distanceReserveKm, availableRangeKm, potentialStops = [] }) {
  if (!isWithinRange) {
    const shortfall = Math.abs(distanceReserveKm);
    return {
      status: 'deficit',
      headline: `Intermediate charge stop required (approx. ${shortfall.toFixed(0)} km deficit).`,
      needsStop: true,
      hasReachableCandidate: potentialStops.length > 0,
      candidate: potentialStops.length > 0 ? potentialStops[0] : null,
      count: potentialStops.length,
      detail: potentialStops.length > 0
        ? `${potentialStops.length} approximate reachable mapped stops identified near route.`
        : 'No reachable charging candidate identified along route. Check external networks before departure.'
    };
  }

  if (distanceReserveKm < availableRangeKm * 0.15) {
    return {
      status: 'low_margin',
      headline: `Low range margin (+${distanceReserveKm.toFixed(0)} km reserve). An intermediate charge stop is prudent.`,
      needsStop: true,
      hasReachableCandidate: potentialStops.length > 0,
      candidate: potentialStops.length > 0 ? potentialStops[0] : null,
      count: potentialStops.length,
      detail: 'Consider a top-up charge along the route.'
    };
  }

  return {
    status: 'comfortable',
    headline: `Comfortable range margin (+${distanceReserveKm.toFixed(0)} km reserve).`,
    needsStop: false,
    hasReachableCandidate: false,
    candidate: null,
    count: 0,
    detail: 'Intermediate charging stop is not strictly necessary under current planning assumptions.'
  };
}
