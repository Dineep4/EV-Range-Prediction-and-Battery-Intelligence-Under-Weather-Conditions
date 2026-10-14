/**
 * VoltWise — Smart Journey Recommendations Logic
 * Rule-based, deterministic EV driving precautions based on verified route,
 * planning parameters, and available ambient weather data.
 *
 * NOTE: These are advisory guidelines based on physics heuristics and driving best practices.
 * They are NOT machine-learning predictions and do NOT override traffic laws or vehicle warnings.
 */

export const WEATHER_THRESHOLDS = {
  HOT_TEMP_C: 30,        // Ambient temperature above which high-heat precautions apply
  COLD_TEMP_C: 10,       // Ambient temperature below which cold-battery precautions apply
  STRONG_WIND_KMH: 25,   // Wind speed above which aerodynamic drag significantly increases
  RAIN_PRECIPITATION_MM: 0.1 // Precipitation threshold indicating wet or rainy road conditions
};

export function classifyWeatherCondition(weather) {
  if (!weather || !Number.isFinite(Number(weather.temperature))) {
    return {
      isAvailable: false,
      isHot: false,
      isCold: false,
      isRain: false,
      isWindy: false,
      isModerate: false,
      temperatureC: null,
      windSpeedKmh: null,
      precipitationMm: null,
      humidityPercent: null
    };
  }

  const temp = Number(weather.temperature);
  const wind = Number.isFinite(Number(weather.wind)) ? Number(weather.wind) : null;
  const precip = Number.isFinite(Number(weather.precipitation)) ? Number(weather.precipitation) : null;
  const humidity = Number.isFinite(Number(weather.humidity)) ? Number(weather.humidity) : null;

  const isHot = temp >= WEATHER_THRESHOLDS.HOT_TEMP_C;
  const isCold = temp <= WEATHER_THRESHOLDS.COLD_TEMP_C;
  const isRain = precip !== null && precip >= WEATHER_THRESHOLDS.RAIN_PRECIPITATION_MM;
  const isWindy = wind !== null && wind >= WEATHER_THRESHOLDS.STRONG_WIND_KMH;
  const isModerate = !isHot && !isCold && !isRain && !isWindy;

  return {
    isAvailable: true,
    isHot,
    isCold,
    isRain,
    isWindy,
    isModerate,
    temperatureC: temp,
    windSpeedKmh: wind,
    precipitationMm: precip,
    humidityPercent: humidity
  };
}

export function buildWeatherPrecautions(weather) {
  const c = classifyWeatherCondition(weather);
  const precautions = [];

  if (!c.isAvailable) {
    precautions.push({
      id: 'weather-unavailable',
      title: 'Weather data unavailable',
      detail: 'Weather data unavailable: Live ambient weather could not be retrieved for this route. Exercise standard EV driving prudence, observe current road conditions, and consult your vehicle dashboard.',
      severity: 'neutral'
    });
    return precautions;
  }

  if (c.isHot) {
    precautions.push({
      id: 'hot-speed',
      title: 'High ambient temperature caution',
      detail: `Current ambient reading is ${c.temperatureC.toFixed(1)}°C (threshold: ≥${WEATHER_THRESHOLDS.HOT_TEMP_C}°C). Avoid unnecessary high-speed driving and strictly follow posted speed limits, as combined high speeds and heat increase thermal load and energy consumption.`,
      severity: 'warning'
    });
    precautions.push({
      id: 'hot-cooling',
      title: 'Sensible cabin cooling & pre-cooling',
      detail: 'Use air conditioning sensibly. If supported by your vehicle, pre-cool the cabin while connected to a charger before departure to save battery range.',
      severity: 'info'
    });
    precautions.push({
      id: 'hot-parking',
      title: 'Park in shaded or covered areas',
      detail: 'Park under shade, carports, or underground parking where possible to minimize solar heat buildup in the cabin and battery pack.',
      severity: 'info'
    });
    precautions.push({
      id: 'hot-safety-critical',
      title: 'Life safety warning',
      detail: 'NEVER leave children, vulnerable individuals, or pets in a parked vehicle, even for a short period. Cabin temperatures can rise to lethal levels rapidly in direct sunlight.',
      severity: 'critical'
    });
    precautions.push({
      id: 'hot-alerts',
      title: 'Monitor vehicle temperature alerts',
      detail: 'Check vehicle dashboard indicators for battery or thermal management alerts and always follow vehicle manufacturer guidance.',
      severity: 'neutral'
    });
  }

  if (c.isCold) {
    precautions.push({
      id: 'cold-efficiency',
      title: 'Cold ambient temperature impact',
      detail: `Current ambient reading is ${c.temperatureC.toFixed(1)}°C (threshold: ≤${WEATHER_THRESHOLDS.COLD_TEMP_C}°C). Cold temperatures reduce electrochemical efficiency and increase heating demand, which can noticeably lower real-world range.`,
      severity: 'warning'
    });
    precautions.push({
      id: 'cold-precondition',
      title: 'Precondition while plugged in',
      detail: 'If supported by your EV, precondition the cabin and battery pack while still connected to a wallbox or charger prior to departure.',
      severity: 'info'
    });
    precautions.push({
      id: 'cold-reserve',
      title: 'Plan a generous range buffer',
      detail: 'Maintain an extra 15–20% range reserve over your planned distance to account for elevated cabin heating and cold tyre rolling resistance.',
      severity: 'info'
    });
    precautions.push({
      id: 'cold-driving',
      title: 'Drive smoothly and avoid rapid acceleration',
      detail: 'Accelerate smoothly until the battery reaches nominal operating temperature, as cold cells have higher internal resistance.',
      severity: 'neutral'
    });
  }

  if (c.isRain) {
    precautions.push({
      id: 'rain-safety',
      title: 'Wet road driving precautions',
      detail: `Precipitation detected (${c.precipitationMm.toFixed(1)} mm). Reduce speed appropriately, increase following distance, and avoid abrupt steering, acceleration, or heavy braking.`,
      severity: 'warning'
    });
    precautions.push({
      id: 'rain-standing-water',
      title: 'Standing water awareness',
      detail: 'Watch for standing water and aquaplaning risks on highway lanes. Water displacement increases tyre rolling resistance and slightly raises energy consumption.',
      severity: 'info'
    });
  }

  if (c.isWindy) {
    precautions.push({
      id: 'wind-aerodynamics',
      title: 'Aerodynamic wind resistance',
      detail: `Elevated wind speed reported (${c.windSpeedKmh.toFixed(0)} km/h; threshold: ≥${WEATHER_THRESHOLDS.STRONG_WIND_KMH} km/h). Headwinds and strong crosswinds significantly increase aerodynamic drag; allow additional range margin.`,
      severity: 'info'
    });
  }

  if (c.isModerate) {
    precautions.push({
      id: 'moderate-ideal',
      title: 'Favorable ambient conditions',
      detail: `Current ambient weather (${c.temperatureC.toFixed(1)}°C) is within a mild, optimal operating bracket. Normal efficient driving techniques will yield near-reference range.`,
      severity: 'positive'
    });
  }

  return precautions;
}

export function buildSpeedAndDrivingGuidance(weather, route) {
  const c = classifyWeatherCondition(weather);
  const items = [];

  items.push({
    title: 'Adhere strictly to posted speed limits',
    text: 'Always respect local speed regulations and prevailing traffic flow. Traffic safety and law compliance must never be compromised for range efficiency.'
  });

  items.push({
    title: 'Maintain smooth acceleration and deceleration',
    text: 'Gentle throttle input reduces peak power draw and preserves battery charge. Anticipate traffic lights and slowdowns to maximize smooth deceleration.'
  });

  if (c.isHot || c.isCold || c.isWindy) {
    items.push({
      title: 'Adopt a conservative driving style in demanding weather',
      text: `In current conditions (${c.isHot ? 'heat' : c.isCold ? 'cold' : 'strong wind'}), aerodynamic drag and thermal systems consume higher power. Cruising at moderate highway speeds rather than the upper limit delivers substantial range savings.`
    });
  } else {
    items.push({
      title: 'Maintain steady highway cruising',
      text: 'Using cruise control on flat, dry highway stretches avoids minor fluctuations that waste energy.'
    });
  }

  items.push({
    title: 'No synthetic speed prescription',
    text: 'VoltWise does NOT prescribe an artificial "optimal speed" from temperature alone. Adjust your speed according to visible road hazards, weather, and legal limits.'
  });

  return items;
}

export function buildBatteryAndLoadPrecautions(soc, capacity, consumption) {
  const items = [];

  items.push({
    title: 'Minimize unnecessary vehicle cargo',
    text: 'Extra payload directly increases rolling resistance and inertial demand during acceleration. Remove heavy cargo or roof boxes when not in use.'
  });

  items.push({
    title: 'Maintain a practical reserve margin',
    text: 'Avoid planning trips to 0% SOC. Aim to arrive with at least 10–15% battery reserve to accommodate unexpected detours, road closures, or busy chargers.'
  });

  items.push({
    title: 'Verify manufacturer tyre pressure specifications',
    text: 'Properly inflated tyres reduce rolling resistance and ensure safe wet-weather braking. Check pressures cold according to the vehicle door-jamb placard.'
  });

  items.push({
    title: 'Cabin HVAC energy awareness',
    text: 'Extreme heater or AC settings can consume 1–4 kW of continuous power. Use seat heaters instead of full cabin heat when driving alone in cool weather.'
  });

  items.push({
    title: 'Rely on instrument cluster alerts',
    text: 'This software uses mathematical planning heuristics and does NOT measure actual real-time battery internal temperature, tyre pressure, or cargo weight. Always rely on certified vehicle instrumentation.'
  });

  return items;
}

export function buildParkingRecommendations(weather) {
  const c = classifyWeatherCondition(weather);
  const items = [];

  if (c.isHot) {
    items.push({
      title: 'Shaded or covered parking strongly recommended',
      text: 'Seek shaded bays, covered multi-story lots, or carports to shield the cabin and high-voltage battery from direct solar radiance and thermal stress.'
    });
  } else if (c.isCold) {
    items.push({
      title: 'Protected parking in freezing conditions',
      text: 'If parking for extended periods in freezing weather, park in enclosed garages where possible and keep the EV plugged into a wallbox so battery thermal management can operate from mains electricity.'
    });
  } else {
    items.push({
      title: 'Standard parking precautions',
      text: 'Park in well-lit, secure bays with easy access to Level 2 or DC charging infrastructure if overnight charging is planned.'
    });
  }

  if (c.isRain) {
    items.push({
      title: 'Avoid low-lying flood-prone parking areas',
      text: 'During heavy rain or drainage overflow risks, avoid parking in low-lying depressions or flood zones. Never drive through standing water of unknown depth.'
    });
  }

  items.push({
    title: 'Advisory distinction',
    text: 'Parking guidance reflects practical automotive care and does not constitute official localized emergency weather alerts. Follow civil protection announcements.'
  });

  return items;
}

export const DEFAULT_CHECKLIST_ITEMS = [
  { id: 'charge_confirmed', label: 'Sufficient charge & practical reserve confirmed for route' },
  { id: 'tyres_checked', label: 'Tyre pressure & instrument cluster warnings verified' },
  { id: 'cable_packed', label: 'Charging cable & compatible network adapters onboard' },
  { id: 'weather_reviewed', label: 'Weather & live route traffic reviewed before departure' },
  { id: 'charging_identified', label: 'Reachable charging stations identified along the route' }
];
