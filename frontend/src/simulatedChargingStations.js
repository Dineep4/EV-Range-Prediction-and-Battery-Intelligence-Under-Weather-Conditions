/**
 * VoltWise Simulated EV Charging Stations Module
 * 
 * Generates 3-4 realistic, geographically plausible, and deterministic simulated EV charging
 * points around any valid source location, with optional route and battery-range safety reserve
 * recommendations.
 * 
 * Strict academic and demo integrity:
 * All stations are clearly identified as simulated demonstration points.
 * No real company names, real-time availability, or fabricated pricing are used.
 */

// Approved professional simulated station names
export const SIMULATED_STATION_NAMES = [
  'VoltWay EV Hub',
  'GreenCharge Station',
  'PowerUp EV Point',
  'E-Motion Charging Hub',
  'SparkCharge Station',
  'UrbanVolt Charging Point',
  'EcoDrive EV Hub',
  'RapidCharge Point',
];

export const SIMULATED_DISCLAIMER =
  'Simulated charging location for demonstration purposes. Not a verified operational charging station.';

export const DEMO_MODE_BANNER =
  'Demo mode: these charging points are simulated for route-planning demonstrations. They are not verified real charging stations.';

// Earth's mean radius in meters
const EARTH_RADIUS_M = 6371000;

/**
 * Deterministic 32-bit hash of coordinates to seed pseudo-random generation.
 * Ensures map does not jump or regenerate randomly for the exact same source point.
 */
export function hashCoordinates(lat, lon) {
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return 42;
  const str = `${Number(lat).toFixed(4)},${Number(lon).toFixed(4)}`;
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

/**
 * Simple deterministic Mulberry32 PRNG.
 */
export function createPrng(seed) {
  let s = seed >>> 0;
  return function next() {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Spherical destination calculation given a starting point, distance (meters), and bearing (degrees).
 */
export function calculateDestinationPoint(latDeg, lonDeg, distanceM, bearingDeg) {
  const dByR = distanceM / EARTH_RADIUS_M;
  const bearingRad = (bearingDeg * Math.PI) / 180;
  const latRad = (latDeg * Math.PI) / 180;
  const lonRad = (lonDeg * Math.PI) / 180;

  const sinLat = Math.sin(latRad) * Math.cos(dByR) + Math.cos(latRad) * Math.sin(dByR) * Math.cos(bearingRad);
  const destLatRad = Math.asin(Math.max(-1, Math.min(1, sinLat)));

  const y = Math.sin(bearingRad) * Math.sin(dByR) * Math.cos(latRad);
  const x = Math.cos(dByR) - Math.sin(latRad) * Math.sin(destLatRad);
  const destLonRad = lonRad + Math.atan2(y, x);

  const destLat = (destLatRad * 180) / Math.PI;
  // Normalize lon to [-180, 180]
  let destLon = (destLonRad * 180) / Math.PI;
  destLon = ((destLon + 540) % 360) - 180;

  return {
    latitude: Math.round(destLat * 100000) / 100000,
    longitude: Math.round(destLon * 100000) / 100000,
  };
}

/**
 * Haversine straight-line distance in kilometers.
 */
export function haversineDistanceKm(lat1, lon1, lat2, lon2) {
  if (!Number.isFinite(lat1) || !Number.isFinite(lon1) || !Number.isFinite(lat2) || !Number.isFinite(lon2)) {
    return 0;
  }
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(Math.max(0, 1 - a)));
  return (EARTH_RADIUS_M * c) / 1000;
}

/**
 * Samples a point along a route polyline at a fraction of distance (0.0 to 1.0).
 */
export function interpolateRoutePoint(points, fraction) {
  if (!Array.isArray(points) || points.length === 0) return null;
  if (points.length === 1 || fraction <= 0) {
    return { latitude: points[0][0], longitude: points[0][1] };
  }
  if (fraction >= 1) {
    const last = points[points.length - 1];
    return { latitude: last[0], longitude: last[1] };
  }

  // Calculate cumulative distances
  const cumDist = [0];
  for (let i = 1; i < points.length; i++) {
    const d = haversineDistanceKm(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1]);
    cumDist.push(cumDist[i - 1] + d);
  }
  const total = cumDist[cumDist.length - 1];
  if (total <= 0) {
    return { latitude: points[0][0], longitude: points[0][1] };
  }

  const target = total * fraction;
  for (let i = 1; i < cumDist.length; i++) {
    if (cumDist[i] >= target) {
      const segSpan = cumDist[i] - cumDist[i - 1];
      const segFrac = segSpan > 0 ? (target - cumDist[i - 1]) / segSpan : 0;
      const lat = points[i - 1][0] + (points[i][0] - points[i - 1][0]) * segFrac;
      const lon = points[i - 1][1] + (points[i][1] - points[i - 1][1]) * segFrac;
      return { latitude: lat, longitude: lon };
    }
  }

  const last = points[points.length - 1];
  return { latitude: last[0], longitude: last[1] };
}

export function isValidCoordinate(lat, lon) {
  if (lat === null || lat === undefined || lat === '' || lon === null || lon === undefined || lon === '') {
    return false;
  }
  const numLat = Number(lat);
  const numLon = Number(lon);
  return (
    Number.isFinite(numLat) &&
    Number.isFinite(numLon) &&
    numLat >= -90 &&
    numLat <= 90 &&
    numLon >= -180 &&
    numLon <= 180
  );
}

/**
 * Resolves source location with prioritized fallback.
 * Priority:
 * 1. Current Search / Source Location (if valid)
 * 2. Origin supplied by Trip Planner (route.start)
 * 3. Map selected coordinates
 * 4. Clearly labelled default location (Hyderabad center: 17.4474, 78.3488)
 */
export function resolveSourceLocation({
  searchLat,
  searchLon,
  searchLabel,
  routeStart,
  routeStartLabel,
  mapSelected,
  defaultLocation = { latitude: 17.4474, longitude: 78.3488, label: 'Default Location: Hyderabad (17.4474, 78.3488)' },
}) {
  // 1. Current Charging Stations search coordinates
  if (isValidCoordinate(searchLat, searchLon)) {
    return {
      latitude: Number(searchLat),
      longitude: Number(searchLon),
      label: searchLabel || `Coordinates (${Number(searchLat).toFixed(4)}, ${Number(searchLon).toFixed(4)})`,
      sourceType: 'search_input',
    };
  }

  // 2. Origin supplied by Trip Planner
  if (Array.isArray(routeStart) && isValidCoordinate(routeStart[0], routeStart[1])) {
    return {
      latitude: Number(routeStart[0]),
      longitude: Number(routeStart[1]),
      label: routeStartLabel || `Trip Planner Origin (${Number(routeStart[0]).toFixed(4)}, ${Number(routeStart[1]).toFixed(4)})`,
      sourceType: 'trip_origin',
    };
  }
  if (routeStart && isValidCoordinate(routeStart.latitude, routeStart.longitude)) {
    return {
      latitude: Number(routeStart.latitude),
      longitude: Number(routeStart.longitude),
      label: routeStart.label || `Trip Planner Origin (${Number(routeStart.latitude).toFixed(4)}, ${Number(routeStart.longitude).toFixed(4)})`,
      sourceType: 'trip_origin',
    };
  }

  // 3. User's chosen map location
  if (mapSelected && isValidCoordinate(mapSelected.latitude, mapSelected.longitude)) {
    return {
      latitude: Number(mapSelected.latitude),
      longitude: Number(mapSelected.longitude),
      label: mapSelected.label || `Map Selection (${Number(mapSelected.latitude).toFixed(4)}, ${Number(mapSelected.longitude).toFixed(4)})`,
      sourceType: 'map_selection',
    };
  }

  // 4. Clearly labelled default location
  return {
    latitude: defaultLocation.latitude,
    longitude: defaultLocation.longitude,
    label: defaultLocation.label,
    sourceType: 'default_fallback',
  };
}

/**
 * Generates 3-4 simulated EV charging points around a source location.
 *
 * @param {Object} options
 * @param {number} options.sourceLat - Latitude of source location
 * @param {number} options.sourceLon - Longitude of source location
 * @param {number} [options.searchRadiusM=8000] - Search radius in meters (default 8 km)
 * @param {number} [options.count=4] - Number of points to generate (3 or 4)
 * @param {Object|null} [options.route] - Optional active route object with distanceKm and points
 * @param {number|null} [options.availableRangeKm] - Estimated vehicle driving range
 * @returns {Array<Object>} List of 3-4 simulated station objects
 */
export function generateSimulatedChargingStations({
  sourceLat,
  sourceLon,
  searchRadiusM = 8000,
  count = 4,
  route = null,
  availableRangeKm = null,
}) {
  if (!isValidCoordinate(sourceLat, sourceLon)) {
    return [];
  }

  const lat = Number(sourceLat);
  const lon = Number(sourceLon);

  const pointCount = Math.max(3, Math.min(4, Math.round(Number(count) || 4)));
  const maxRadiusM = Math.max(2500, Math.min(25000, Number(searchRadiusM) || 8000));
  const maxRadiusKm = maxRadiusM / 1000;

  // Deterministic seed
  const seed = hashCoordinates(lat, lon);
  const rand = createPrng(seed);

  // Shuffle station names deterministically to pick `pointCount` distinct names
  const availableNames = [...SIMULATED_STATION_NAMES];
  for (let i = availableNames.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [availableNames[i], availableNames[j]] = [availableNames[j], availableNames[i]];
  }
  const chosenNames = availableNames.slice(0, pointCount);

  // Hardware templates (Type 2, CCS2, varied powers)
  const hardwareProfiles = [
    {
      connectors: { 'Type 2': 2, 'CCS2': 2 },
      chargingPower: '50 kW Fast Charger',
      powerKw: 50,
      connectorTypes: 'Type 2 · CCS2',
    },
    {
      connectors: { 'Type 2': 2 },
      chargingPower: '22 kW AC Destination',
      powerKw: 22,
      connectorTypes: 'Type 2',
    },
    {
      connectors: { 'CCS2': 4 },
      chargingPower: '60 kW DC Rapid',
      powerKw: 60,
      connectorTypes: 'CCS2',
    },
    {
      connectors: { 'Type 2': 2, 'CCS2': 4 },
      chargingPower: '120 kW High-Power Hub',
      powerKw: 120,
      connectorTypes: 'Type 2 · CCS2',
    },
  ];

  const hasRouteGeometry =
    route &&
    Array.isArray(route.points) &&
    route.points.length > 1 &&
    Number.isFinite(route.distanceKm) &&
    route.distanceKm > 0;

  const tripDistKm = hasRouteGeometry ? route.distanceKm : null;
  const rangeKm = Number.isFinite(availableRangeKm) && availableRangeKm > 0 ? availableRangeKm : null;

  const stations = [];

  // Base angles distributed evenly around source (e.g. 40°, 130°, 220°, 310°)
  const baseAngleStep = 360 / pointCount;

  for (let i = 0; i < pointCount; i++) {
    const hardware = hardwareProfiles[i % hardwareProfiles.length];
    const stationName = chosenNames[i];

    let destLat = lat;
    let destLon = lon;
    let distKm = 0;
    let routeProgressKm = null;
    let distanceFromRouteKm = null;
    let isReachable = true;
    let recommendationStatus = 'nearby';
    let recommendationBadge = 'Simulated Nearby Point';
    let recommendationReason = 'Straight-line map distance from search center. Road distance may be longer.';

    if (hasRouteGeometry) {
      // Points distributed along route progression
      // E.g. for 4 points: ~20%, 45%, 70%, 90% along the route
      const progressFractions = [0.22, 0.45, 0.68, 0.88];
      const progFrac = progressFractions[i] || (0.2 + i * 0.22);
      routeProgressKm = Math.round(tripDistKm * progFrac * 10) / 10;

      const corridorPoint = interpolateRoutePoint(route.points, progFrac);
      if (corridorPoint) {
        // Small lateral offset perpendicular to corridor: 0.8 km - 2.5 km
        const lateralAngle = (i % 2 === 0 ? 80 : 260) + (rand() * 20 - 10);
        const lateralDistM = 800 + rand() * 1700; // 0.8 - 2.5 km
        const pos = calculateDestinationPoint(corridorPoint.latitude, corridorPoint.longitude, lateralDistM, lateralAngle);
        destLat = pos.latitude;
        destLon = pos.longitude;
        distanceFromRouteKm = Math.round((lateralDistM / 1000) * 10) / 10;
        distKm = Math.round(haversineDistanceKm(lat, lon, destLat, destLon) * 10) / 10;
      } else {
        // Fallback to radial
        const bearing = (i * baseAngleStep + 25 + rand() * 20) % 360;
        const dM = Math.min(maxRadiusM, 2000 + (i + 1) * 2200 + rand() * 1200);
        const pos = calculateDestinationPoint(lat, lon, dM, bearing);
        destLat = pos.latitude;
        destLon = pos.longitude;
        distKm = Math.round((dM / 1000) * 10) / 10;
        distanceFromRouteKm = distKm;
      }

      // Route and Battery Range aware logic
      if (rangeKm !== null) {
        if (tripDistKm > rangeKm) {
          // Trip exceeds available range!
          // A safe stop should occur roughly between 35% and 65% of available range (preserving a safety reserve of at least 25%).
          // Requirement: "For example, if estimated available range is 40 km and the trip distance is 50 km, recommend a simulated stop roughly 15–25 km along the route...
          // For a 40 km available range, do not recommend a first stop 35–40 km away when a safer alternative can be shown."
          const safeReserveThresholdKm = rangeKm * 0.70; // 70% of available range
          const criticalDepletionThresholdKm = rangeKm * 0.95; // 95% of available range

          if (routeProgressKm <= safeReserveThresholdKm && routeProgressKm >= 10) {
            recommendationStatus = 'recommended_stop';
            recommendationBadge = 'Recommended Charge Stop';
            recommendationReason = `Reachable with a safe battery reserve (~${Math.round(rangeKm - routeProgressKm)} km remaining buffer) before range depletion.`;
          } else if (routeProgressKm <= criticalDepletionThresholdKm) {
            recommendationStatus = 'marginal_stop';
            recommendationBadge = 'Marginal / High-Risk Stop';
            recommendationReason = `Close to battery exhaustion (<15% buffer). Earlier stop recommended.`;
          } else if (routeProgressKm > rangeKm) {
            isReachable = false;
            recommendationStatus = 'unreachable';
            recommendationBadge = 'Unreachable Without Prior Charge';
            recommendationReason = `Exceeds current driving range (${Math.round(routeProgressKm)} km > ${Math.round(rangeKm)} km range).`;
          } else {
            recommendationStatus = 'early_stop';
            recommendationBadge = 'Early Stop';
            recommendationReason = `Early route stop; useful for topping up before long highway segments.`;
          }
        } else {
          // Trip is comfortably within available range!
          recommendationStatus = 'optional';
          recommendationBadge = 'Optional Charging Stop';
          recommendationReason = `Trip (${Math.round(tripDistKm)} km) is within available range (${Math.round(rangeKm)} km). Charging is optional.`;
        }
      }
    } else {
      // Standalone search without active route
      // Distribute points across varied distances (typically 2 - 14 km) within search radius
      const bearing = (i * baseAngleStep + 20 + rand() * 25) % 360;
      // Stagger distances: e.g. 2.8 km, 5.5 km, 8.2 km, 11.5 km (bounded by search radius)
      const baseDistanceFraction = 0.25 + (i * 0.22); // 25%, 47%, 69%, 91% of max radius
      const targetDistM = Math.max(2000, Math.min(maxRadiusM * 0.95, maxRadiusM * baseDistanceFraction + rand() * 1000));
      const pos = calculateDestinationPoint(lat, lon, targetDistM, bearing);
      destLat = pos.latitude;
      destLon = pos.longitude;
      distKm = Math.round((targetDistM / 1000) * 10) / 10;
      distanceFromRouteKm = distKm;

      recommendationStatus = 'nearby';
      recommendationBadge = 'Simulated Nearby Point';
      recommendationReason =
        'Straight-line geodesic distance from search center (approximate map discovery only; road distance may be longer).';
    }

    stations.push({
      id: `sim-${seed}-${i + 1}`,
      osmType: 'simulated',
      isSimulated: true,
      name: stationName,
      operator: 'VoltWise Simulated Network (Demo)',
      lat: destLat,
      lon: destLon,
      distanceFromRouteKm: distanceFromRouteKm !== null ? distanceFromRouteKm : distKm,
      distanceFromSourceKm: distKm,
      routeProgressKm,
      connectors: hardware.connectors,
      connectorTypes: hardware.connectorTypes,
      chargingPower: hardware.chargingPower,
      powerKw: hardware.powerKw,
      access: 'Public simulated access',
      openingHours: '24/7 (Simulated)',
      fee: 'Simulated pricing · Demo purposes only',
      statusLabel: 'Simulated location',
      disclaimer: SIMULATED_DISCLAIMER,
      recommendationStatus,
      recommendationBadge,
      recommendationReason,
      isReachable,
      mapUrl: `https://www.openstreetmap.org/#map=16/${destLat}/${destLon}`,
    });
  }

  // Sort by route progress if route exists, else by distance from source
  if (hasRouteGeometry) {
    stations.sort((a, b) => (a.routeProgressKm || 0) - (b.routeProgressKm || 0));
  } else {
    stations.sort((a, b) => a.distanceFromSourceKm - b.distanceFromSourceKm);
  }

  return stations;
}
