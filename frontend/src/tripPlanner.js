export const EPA_FEATURE_ORDER = [
  'year',
  'city_e_kwh_100_miles',
  'highway_e_kwh_100_miles',
  'combined_e_kwh_100_miles',
  'charge_240_hours',
  'drive',
  'vehicle_class',
];

export function resolveEpaVehicleSelection(value, vehicles) {
  const match = String(value).trim().match(/^(.*?)\s*·\s*record\s+(\d+)\s*$/i);
  if (!match) return null;
  const id = Number(match[2]);
  const vehicle = vehicles.find((item) => Number(item.source_record_id) === id);
  if (!vehicle || match[1].trim() !== String(vehicle.label).trim()) return null;
  if (!vehicle.features || EPA_FEATURE_ORDER.some((key) => {
    const feature = vehicle.features[key];
    return feature === null || feature === undefined || feature === '';
  })) return null;
  if (!Number.isSafeInteger(id) || id <= 0) return null;
  return vehicle;
}

export function formatRouteDuration(durationSeconds) {
  const seconds = Number(durationSeconds);
  if (!Number.isFinite(seconds) || seconds < 0) return 'Unavailable';
  const totalMinutes = Math.round(seconds / 60);
  if (totalMinutes < 60) return totalMinutes < 1 ? '<1 min' : `${totalMinutes} min`;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return minutes ? `${hours} hr ${minutes} min` : `${hours} hr`;
}

export function sampleRoutePoints(points, maxPoints = 8) {
  if (points.length <= maxPoints) return points;
  return Array.from({ length: maxPoints }, (_, index) => (
    points[Math.round(index * (points.length - 1) / (maxPoints - 1))]
  ));
}

export function recommendPotentialChargeStops(stations, availableRangeKm) {
  const rangeKm = Number(availableRangeKm);
  if (!Number.isFinite(rangeKm) || rangeKm <= 0) return [];
  return stations
    .filter((station) => (
      Number.isFinite(station.routeProgressKm)
      && Number.isFinite(station.distanceFromRouteKm)
      && station.distanceFromRouteKm <= 4.5
      && station.routeProgressKm + station.distanceFromRouteKm <= rangeKm
    ))
    .sort((first, second) => first.routeProgressKm - second.routeProgressKm);
}

export function handleTripFormSubmit(event, executePlan) {
  if (event && typeof event.preventDefault === 'function') {
    event.preventDefault();
  }
  if (typeof executePlan === 'function') {
    return executePlan(event);
  }
  return null;
}

export function validateTripPlanInputs({ start, destination } = {}) {
  const cleanStart = String(start || '').trim();
  const cleanDest = String(destination || '').trim();
  if (!cleanStart || !cleanDest) {
    throw new Error('Enter both a starting point and a destination.');
  }
  if (cleanStart.toLowerCase() === cleanDest.toLowerCase()) {
    throw new Error('Choose different starting and destination locations.');
  }
  return { start: cleanStart, destination: cleanDest };
}

export function buildRouteResult({
  routeData,
  physics,
  epaResult = null,
  tripWeatherFactor = 1,
  rangeSource = 'FastAPI physics-based estimate',
} = {}) {
  if (
    !routeData
    || !Array.isArray(routeData.geometry)
    || routeData.geometry.length < 2
    || !Number.isFinite(routeData.distance_m)
    || routeData.distance_m <= 0
    || !Number.isFinite(routeData.duration_s)
    || routeData.duration_s < 0
  ) {
    throw new Error('Routing provider returned incomplete route distance, duration, or geometry.');
  }
  const points = routeData.geometry.map((point) => [
    Number(point.latitude),
    Number(point.longitude),
  ]);
  if (points.some((point) => point.length !== 2 || !point.every(Number.isFinite))) {
    throw new Error('Routing provider returned invalid route coordinates.');
  }
  return {
    start: [routeData.start?.latitude, routeData.start?.longitude],
    startLabel: routeData.start?.label || 'Trip start',
    destination: [routeData.destination?.latitude, routeData.destination?.longitude],
    destinationLabel: routeData.destination?.label || 'Destination',
    points,
    distanceKm: routeData.distance_m / 1000,
    durationSeconds: routeData.duration_s,
    epaPrediction: epaResult,
    availableRangeKm: physics?.estimatedRangeKm ?? 0,
    usableEnergyKwh: physics?.usableEnergyKwh ?? 0,
    unadjustedRangeKm: physics?.unadjustedRangeKm ?? 0,
    weatherFactor: tripWeatherFactor,
    weatherAdjusted: tripWeatherFactor !== 1,
    rangeSource,
    routeSource: routeData.source || 'OSRM routing',
  };
}
