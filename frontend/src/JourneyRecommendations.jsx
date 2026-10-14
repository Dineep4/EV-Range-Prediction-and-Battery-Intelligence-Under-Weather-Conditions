import React, { useState } from 'react';
import {
  AlertTriangle,
  BatteryCharging,
  CarFront,
  CheckCircle2,
  CheckSquare,
  Clock,
  CloudRain,
  CloudSun,
  Compass,
  Gauge,
  Info,
  Navigation,
  Route,
  ShieldAlert,
  ShieldCheck,
  Square,
  Thermometer,
  Wind,
  Zap
} from 'lucide-react';
import {
  buildBatteryAndLoadPrecautions,
  buildParkingRecommendations,
  buildSpeedAndDrivingGuidance,
  classifyWeatherCondition,
  DEFAULT_CHECKLIST_ITEMS
} from './journeyRecommendations.js';
import { buildJourneyWeatherPrecautions } from './weatherAdvice.js';
import { formatRouteDuration } from './tripPlanner.js';

const fmt = (n, d = 1) =>
  n === null || n === undefined || n === '' || !Number.isFinite(Number(n))
    ? '—'
    : Number(n).toFixed(d);

export function JourneyRecommendations({
  route,
  weather = null,
  departureWeather = null,
  destinationWeather = null,
  weatherState = 'unavailable',
  soc = 80,
  soh = 92,
  capacity = 60,
  consumption = 165,
  epaVehicle = null
}) {
  if (!route) return null;

  const [checkedItems, setCheckedItems] = useState({
    charge_confirmed: false,
    tyres_checked: false,
    cable_packed: false,
    weather_reviewed: false,
    charging_identified: false
  });

  const toggleCheck = (id) => {
    setCheckedItems((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const completedCount = Object.values(checkedItems).filter(Boolean).length;

  const depWeather = departureWeather || weather;
  const destWeather = destinationWeather;

  const journeyAnalysis = buildJourneyWeatherPrecautions(depWeather, destWeather, route);
  const speedGuidance = buildSpeedAndDrivingGuidance(depWeather, route);
  const batteryPrecautions = buildBatteryAndLoadPrecautions(soc, capacity, consumption);

  const remainingKm = route.availableRangeKm - route.distanceKm;

  return (
    <section className="panel journey-recommendations-section" style={{ marginTop: '22px' }}>
      {/* Section Header */}
      <div className="panel-head" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '16px', marginBottom: '22px' }}>
        <div>
          <div className="eyebrow">INTELLIGENT ROUTE ADVISORY</div>
          <h2 style={{ margin: '4px 0 0', fontFamily: 'var(--font-heading)', fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)' }}>
            Journey Weather &amp; Route Precautions
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: '13px', color: 'var(--text-secondary)' }}>
            Context-specific route weather observations and precautions for your calculated electric journey.
          </p>
        </div>
        <span className="soft-badge" style={{ background: 'var(--accent-emerald-dim)', color: 'var(--accent-emerald)', borderColor: 'var(--accent-emerald-border)' }}>
          JOURNEY ADVISORY
        </span>
      </div>

      {/* A. Journey Overview */}
      <div className="recommendations-subblock">
        <div className="subblock-header">
          <div className="subblock-title">
            <Compass size={18} color="var(--accent-blue)" />
            <h3>A. Selected Journey Overview</h3>
          </div>
          <span className="subblock-tag">PLANNING CONTEXT</span>
        </div>

        <div className="journey-overview-grid">
          <div className="overview-card">
            <span className="overview-label">Route</span>
            <strong className="overview-value" title={`${route.startLabel} to ${route.destinationLabel}`}>
              {route.startLabel?.split(',')[0]} → {route.destinationLabel?.split(',')[0]}
            </strong>
            <small className="overview-sub">{fmt(route.distanceKm, 1)} km road distance</small>
          </div>

          <div className="overview-card">
            <span className="overview-label">Duration &amp; Range</span>
            <strong className="overview-value">
              {formatRouteDuration(route.durationSeconds)} · {fmt(route.availableRangeKm, 0)} km
            </strong>
            <small className="overview-sub">
              {remainingKm >= 0 ? `+${fmt(remainingKm, 0)} km estimated reserve` : `${fmt(Math.abs(remainingKm), 0)} km below route distance`}
            </small>
          </div>

          <div className="overview-card">
            <span className="overview-label">Planning Parameters</span>
            <strong className="overview-value">
              {soc}% SOC · {capacity} kWh
            </strong>
            <small className="overview-sub">
              {soh}% plan health · {consumption} Wh/km
            </small>
          </div>

          <div className="overview-card">
            <span className="overview-label">Usable Energy &amp; Adjustment</span>
            <strong className="overview-value">
              {fmt(route.usableEnergyKwh || (capacity * soc * soh / 10000), 1)} kWh
            </strong>
            <small className="overview-sub">
              {route.weatherAdjusted ? `Weather factor: ${fmt(route.weatherFactor, 3)}` : 'No weather penalty applied'}
            </small>
          </div>
        </div>

        <p className="fine-print" style={{ marginTop: '10px' }}>
          * Values represent deterministic physics calculations derived from entered battery capacity, state of charge, and public routing distance. Real-world completion is not guaranteed.
        </p>
      </div>

      {/* B. Weather Relevant to Selected Journey */}
      <div className="recommendations-subblock" style={{ marginTop: '24px' }}>
        <div className="subblock-header">
          <div className="subblock-title">
            <CloudSun size={18} color="var(--accent-emerald)" />
            <h3>B. Endpoint Weather Relevant to Selected Journey</h3>
          </div>
          <span className="subblock-tag">
            {journeyAnalysis.hasAnyWeather ? 'ENDPOINT OBSERVATIONS' : 'LIMITED ADVISORY'}
          </span>
        </div>

        <div className="journey-overview-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
          {/* Departure Point Weather */}
          <div className="overview-card" style={{ background: 'var(--bg-surface-elevated)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <span className="overview-label" style={{ fontWeight: 700, color: 'var(--accent-blue)' }}>📍 Departure Point</span>
              <span className="soft-badge" style={{ fontSize: '9px', padding: '1px 6px' }}>
                {depWeather ? 'LOCAL OBSERVATION' : 'UNAVAILABLE'}
              </span>
            </div>
            <strong className="overview-value" style={{ fontSize: '15px' }}>
              {route.startLabel?.split(',')[0]}
            </strong>
            {depWeather && Number.isFinite(Number(depWeather.temperature)) ? (
              <div style={{ marginTop: '8px', fontSize: '12px', display: 'grid', gap: '3px', color: 'var(--text-secondary)' }}>
                <span><b>Temperature:</b> {fmt(depWeather.temperature, 1)}°C</span>
                <span><b>Humidity:</b> {depWeather.humidity}% · <b>Wind:</b> {fmt(depWeather.wind, 0)} km/h</span>
                <span><b>Precipitation:</b> {fmt(depWeather.precipitation, 1)} mm</span>
                <small style={{ color: 'var(--text-muted)', marginTop: '4px' }}>Observed: {depWeather.time || 'Live'} ({depWeather.source || 'Open-Meteo'})</small>
              </div>
            ) : (
              <small className="overview-sub" style={{ marginTop: '6px' }}>Departure weather observation unavailable.</small>
            )}
          </div>

          {/* Destination Point Weather */}
          <div className="overview-card" style={{ background: 'var(--bg-surface-elevated)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <span className="overview-label" style={{ fontWeight: 700, color: 'var(--accent-purple)' }}>🏁 Destination Point</span>
              <span className="soft-badge" style={{ fontSize: '9px', padding: '1px 6px' }}>
                {destWeather ? 'LOCAL OBSERVATION' : 'ADVISORY ONLY'}
              </span>
            </div>
            <strong className="overview-value" style={{ fontSize: '15px' }}>
              {route.destinationLabel?.split(',')[0]}
            </strong>
            {destWeather && Number.isFinite(Number(destWeather.temperature)) ? (
              <div style={{ marginTop: '8px', fontSize: '12px', display: 'grid', gap: '3px', color: 'var(--text-secondary)' }}>
                <span><b>Temperature:</b> {fmt(destWeather.temperature, 1)}°C</span>
                <span><b>Humidity:</b> {destWeather.humidity}% · <b>Wind:</b> {fmt(destWeather.wind, 0)} km/h</span>
                <span><b>Precipitation:</b> {fmt(destWeather.precipitation, 1)} mm</span>
                <small style={{ color: 'var(--text-muted)', marginTop: '4px' }}>Observed: {destWeather.time || 'Live'} ({destWeather.source || 'Open-Meteo'})</small>
              </div>
            ) : (
              <div style={{ marginTop: '6px', fontSize: '11.5px', color: 'var(--text-muted)' }}>
                <span>Separate destination observation unavailable. Departure conditions serve as the local planning advisory.</span>
              </div>
            )}
          </div>
        </div>

        {/* Transparency note about forecasts vs observations */}
        <div className="notice" style={{ marginTop: '12px', background: 'var(--bg-surface-elevated)', borderColor: 'var(--border-subtle)', color: 'var(--text-secondary)' }}>
          <Info size={16} color="var(--accent-blue)" />
          <div>
            <b>Weather Source Transparency &amp; Microclimate Limitations:</b>
            <p style={{ margin: '3px 0 0', fontSize: '11.5px', lineHeight: '1.5' }}>
              {journeyAnalysis.planningExplanation.limitationNotice}
            </p>
          </div>
        </div>
      </div>

      {/* C. Journey-Specific Precautions */}
      <div className="recommendations-subblock" style={{ marginTop: '24px' }}>
        <div className="subblock-header">
          <div className="subblock-title">
            <AlertTriangle size={18} color="var(--accent-amber)" />
            <h3>C. Journey-Specific Precautions</h3>
          </div>
          <span className="subblock-tag">ROUTE CONDITIONS</span>
        </div>

        <div className="recommendation-cards-grid">
          {journeyAnalysis.precautions.map((item) => (
            <div
              key={item.id}
              className={`guidance-card ${
                item.severity === 'warning'
                  ? 'guidance-warning'
                  : item.severity === 'positive'
                  ? 'guidance-positive'
                  : ''
              }`}
            >
              <div className="guidance-card-head">
                {item.severity === 'warning' ? (
                  <AlertTriangle size={17} color="var(--accent-amber)" />
                ) : item.severity === 'positive' ? (
                  <CheckCircle2 size={17} color="var(--accent-emerald)" />
                ) : (
                  <Info size={17} color="var(--accent-blue)" />
                )}
                <strong>{item.title}</strong>
              </div>
              <p>{item.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* D. Transparent Explanation of Weather & Range Planning */}
      <div className="recommendations-subblock" style={{ marginTop: '24px' }}>
        <div className="subblock-header">
          <div className="subblock-title">
            <Zap size={18} color="var(--accent-blue)" />
            <h3>D. How Weather May Affect Trip Planning</h3>
          </div>
          <span className="subblock-tag">ENERGY MODELING</span>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div className="guidance-card" style={{ borderLeft: '3px solid var(--accent-blue)' }}>
            <div className="guidance-card-head">
              <Thermometer size={16} color="var(--accent-blue)" />
              <strong>Temperature &amp; HVAC Loads</strong>
            </div>
            <p>
              Ambient temperature influences lithium-ion cell electrochemical reaction rates and cabin climate demand. Heating in cold weather typically draws 2–5 kW from high-voltage traction packs, reducing range. Cooling in warm weather uses 1–3 kW.
            </p>
          </div>

          <div className="guidance-card" style={{ borderLeft: '3px solid var(--accent-amber)' }}>
            <div className="guidance-card-head">
              <CloudRain size={16} color="var(--accent-amber)" />
              <strong>Precipitation &amp; Aerodynamic Drag</strong>
            </div>
            <p>
              Wet asphalt forces tyres to displace water, elevating rolling resistance by 5–15%. Strong headwinds increase aerodynamic drag proportionally to the square of relative air speed. Cruising at moderate highway speeds mitigates both effects.
            </p>
          </div>

          <div className="guidance-card" style={{ borderLeft: '3px solid var(--accent-emerald)' }}>
            <div className="guidance-card-head">
              <ShieldCheck size={16} color="var(--accent-emerald)" />
              <strong>Physics Range Baseline &amp; Adjustment Heuristic</strong>
            </div>
            <p>
              {journeyAnalysis.planningExplanation.physicsFormula} {journeyAnalysis.planningExplanation.weatherAdjustment}
            </p>
          </div>
        </div>
      </div>

      {/* E. Speed & Driving Style */}
      <div className="recommendations-subblock" style={{ marginTop: '24px' }}>
        <div className="subblock-header">
          <div className="subblock-title">
            <Gauge size={18} color="var(--accent-blue)" />
            <h3>E. Speed &amp; Safe Driving Guidance</h3>
          </div>
          <span className="subblock-tag">SAFETY &amp; EFFICIENCY</span>
        </div>

        <div className="recommendation-cards-grid">
          {speedGuidance.map((item, idx) => (
            <div key={idx} className="guidance-card">
              <div className="guidance-card-head">
                <CheckCircle2 size={16} color="var(--accent-emerald)" />
                <strong>{item.title}</strong>
              </div>
              <p>{item.text}</p>
            </div>
          ))}
        </div>
      </div>

      {/* F. Journey Readiness Checklist */}
      <div className="recommendations-subblock" style={{ marginTop: '24px' }}>
        <div className="subblock-header">
          <div className="subblock-title">
            <CheckSquare size={18} color="var(--accent-emerald)" />
            <h3>F. Pre-Departure Verification Checklist</h3>
          </div>
          <span className="subblock-tag" style={{ background: completedCount === DEFAULT_CHECKLIST_ITEMS.length ? 'var(--accent-emerald-dim)' : 'var(--bg-surface-elevated)', color: completedCount === DEFAULT_CHECKLIST_ITEMS.length ? 'var(--accent-emerald)' : 'var(--text-secondary)' }}>
            {completedCount} OF {DEFAULT_CHECKLIST_ITEMS.length} VERIFIED
          </span>
        </div>

        <div className="checklist-container">
          {DEFAULT_CHECKLIST_ITEMS.map((item) => {
            const isChecked = Boolean(checkedItems[item.id]);
            return (
              <label key={item.id} className={`checklist-item ${isChecked ? 'checked' : ''}`}>
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => toggleCheck(item.id)}
                  style={{ display: 'none' }}
                />
                <span className="checklist-box">
                  {isChecked ? <CheckSquare size={18} color="var(--accent-emerald)" /> : <Square size={18} color="var(--border-medium)" />}
                </span>
                <span className="checklist-label" style={{ textDecoration: isChecked ? 'line-through' : 'none', color: isChecked ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                  {item.label}
                </span>
              </label>
            );
          })}
        </div>

        <div className="notice" style={{ marginTop: '16px', background: 'var(--bg-surface-elevated)', borderColor: 'var(--border-subtle)', color: 'var(--text-secondary)' }}>
          <Info size={16} color="var(--accent-emerald)" />
          <div>
            <b>Pre-Departure Advisory:</b>
            <p style={{ margin: '3px 0 0', fontSize: '11.5px', lineHeight: '1.5' }}>
              These rules are based on practical EV engineering principles and ambient conditions. They do not replace certified vehicle instrumentation, local road signage, or emergency safety alerts. Always drive safely.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
export default JourneyRecommendations;
