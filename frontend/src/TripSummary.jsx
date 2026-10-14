import React from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BatteryCharging,
  CheckCircle2,
  Clock,
  CloudSun,
  Info,
  MapPin,
  Navigation,
  PlugZap,
  Route,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import { formatRouteDuration, recommendPotentialChargeStops } from './tripPlanner.js';
import {
  calculateDistanceReserve,
  calculateArrivalSocPercent,
  buildTripWeatherPrecautions,
  buildTripChargingAdvice
} from './tripSummaryLogic.js';

const fmt = (n, d = 1) =>
  n === null || n === undefined || n === '' || !Number.isFinite(Number(n))
    ? '—'
    : Number(n).toFixed(d);

export function TripSummary({
  route,
  weather = null,
  departureWeather = null,
  destinationWeather = null,
  weatherState = 'unavailable',
  stations = [],
  soc = 80,
  capacity = 60,
  consumption = 165,
  onNavigateToCharging = () => {},
}) {
  if (!route) return null;

  const reserve = calculateDistanceReserve(route.availableRangeKm, route.distanceKm);
  const distanceKm = reserve.distanceKm;
  const availableRangeKm = reserve.availableRangeKm;
  const remainingKm = reserve.distanceReserveKm;
  const isWithinRange = reserve.isWithinRange;

  const arrivalSocPercent = calculateArrivalSocPercent(soc, distanceKm, availableRangeKm);
  const potentialStops = recommendPotentialChargeStops(stations, availableRangeKm);
  const weatherInfo = buildTripWeatherPrecautions(departureWeather || weather, destinationWeather, weatherState);
  const chargingAdvice = buildTripChargingAdvice({
    isWithinRange,
    distanceReserveKm: remainingKm,
    availableRangeKm,
    potentialStops
  });

  return (
    <section className="panel trip-summary-panel" style={{ marginTop: '20px', border: '1px solid var(--border-medium)', boxShadow: 'var(--shadow-md)' }}>
      {/* Header */}
      <div className="panel-head" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '14px', marginBottom: '18px' }}>
        <div>
          <div className="eyebrow" style={{ color: 'var(--accent-emerald)', fontWeight: 800 }}>JOURNEY VERIFICATION</div>
          <h3 style={{ margin: '3px 0 0', fontSize: '20px', fontWeight: 800, color: 'var(--text-primary)' }}>
            Trip Summary
          </h3>
          <p style={{ margin: '3px 0 0', fontSize: '13px', color: 'var(--text-secondary)' }}>
            Consolidated journey metrics, energy reserve, endpoint weather context, and charging evaluation.
          </p>
        </div>
        <span
          className="soft-badge"
          style={{
            background: isWithinRange ? 'var(--accent-emerald-dim)' : 'var(--accent-rose-dim)',
            color: isWithinRange ? 'var(--accent-emerald)' : 'var(--accent-rose)',
            borderColor: isWithinRange ? 'var(--accent-emerald-border)' : 'var(--accent-rose-border)',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            fontWeight: 700
          }}
        >
          {isWithinRange ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
          {isWithinRange ? 'WITHIN ESTIMATED RANGE' : 'RANGE INSUFFICIENT'}
        </span>
      </div>

      {/* 4-Box Summary Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px' }}>

        {/* Box A: Route Details */}
        <div style={{ padding: '14px', background: 'var(--bg-surface-elevated)', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', color: 'var(--accent-blue)' }}>
            <Route size={16} />
            <strong style={{ fontSize: '13.5px', color: 'var(--text-primary)' }}>A. Route Details</strong>
          </div>
          <div style={{ display: 'grid', gap: '6px', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
            <div>
              <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '11px' }}>From</span>
              <strong style={{ color: 'var(--text-primary)' }}>{route.startLabel}</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '11px' }}>To</span>
              <strong style={{ color: 'var(--text-primary)' }}>{route.destinationLabel}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', paddingTop: '6px', borderTop: '1px dashed var(--border-subtle)' }}>
              <span>Road Distance: <b style={{ color: 'var(--text-primary)' }}>{fmt(distanceKm, 1)} km</b></span>
              <span>Driving Time: <b style={{ color: 'var(--text-primary)' }}>{formatRouteDuration(route.durationSeconds)}</b></span>
            </div>
          </div>
        </div>

        {/* Box B: Estimated Range & Reserve */}
        <div style={{ padding: '14px', background: 'var(--bg-surface-elevated)', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', color: isWithinRange ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
            <BatteryCharging size={16} />
            <strong style={{ fontSize: '13.5px', color: 'var(--text-primary)' }}>B. Range &amp; Battery Reserve</strong>
          </div>
          <div style={{ display: 'grid', gap: '6px', fontSize: '12.5px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Available Range:</span>
              <b style={{ color: 'var(--text-primary)' }}>{fmt(availableRangeKm, 0)} km</b>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>{isWithinRange ? 'Distance Reserve:' : 'Range Shortfall:'}</span>
              <b style={{ color: isWithinRange ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                {isWithinRange ? `+${fmt(remainingKm, 1)} km` : `-${fmt(Math.abs(remainingKm), 1)} km`}
              </b>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Arrival Battery Reserve:</span>
              <b style={{ color: isWithinRange ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                {isWithinRange ? `~${arrivalSocPercent}% SOC` : '0% (Depleted before arrival)'}
              </b>
            </div>
            <p style={{ margin: '4px 0 0', fontSize: '11px', color: 'var(--text-muted)', lineHeight: '1.35' }}>
              {isWithinRange
                ? 'Estimated reserve on arrival under entered planning consumption.'
                : 'Trip exceeds available range; an intermediate charge stop is required.'}
            </p>
          </div>
        </div>

        {/* Box C: Weather Precautions */}
        <div style={{ padding: '14px', background: 'var(--bg-surface-elevated)', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', color: 'var(--accent-amber)' }}>
            <CloudSun size={16} />
            <strong style={{ fontSize: '13.5px', color: 'var(--text-primary)' }}>C. Weather Precautions</strong>
          </div>
          <div style={{ fontSize: '12px', lineHeight: '1.4' }}>
            <div style={{
              padding: '6px 9px',
              borderRadius: '6px',
              background: weatherInfo.severity === 'warning' ? 'var(--accent-amber-dim)' : weatherInfo.severity === 'positive' ? 'var(--accent-emerald-dim)' : 'var(--bg-surface)',
              border: `1px solid ${weatherInfo.severity === 'warning' ? 'var(--accent-amber-border)' : weatherInfo.severity === 'positive' ? 'var(--accent-emerald-border)' : 'var(--border-subtle)'}`,
              marginBottom: '6px'
            }}>
              <strong style={{ display: 'block', color: 'var(--text-primary)', fontSize: '12px' }}>
                {weatherInfo.headline}
              </strong>
              <span style={{ color: 'var(--text-secondary)', fontSize: '11.5px' }}>
                {weatherInfo.detail}
              </span>
            </div>
            <span style={{ fontSize: '10.5px', color: 'var(--text-muted)', display: 'block' }}>
              {weatherInfo.disclaimer}
            </span>
          </div>
        </div>

        {/* Box D: Charging-Stop Suggestions */}
        <div style={{ padding: '14px', background: 'var(--bg-surface-elevated)', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', color: 'var(--accent-purple)' }}>
            <PlugZap size={16} />
            <strong style={{ fontSize: '13.5px', color: 'var(--text-primary)' }}>D. Charging Suggestions</strong>
          </div>
          <div style={{ fontSize: '12px', lineHeight: '1.4' }}>
            {chargingAdvice.status === 'deficit' ? (
              <>
                <p style={{ margin: '0 0 6px', color: 'var(--accent-rose)', fontWeight: 600 }}>
                  {chargingAdvice.headline}
                </p>
                {chargingAdvice.hasReachableCandidate ? (
                  <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                    <span>{chargingAdvice.detail}</span>
                    <strong style={{ display: 'block', color: 'var(--text-primary)', marginTop: '2px' }}>
                      {chargingAdvice.candidate.name || 'Mapped Charger'}
                      {chargingAdvice.candidate.isSimulated && <small style={{ color: 'var(--accent-purple)', marginLeft: '4px' }}>(Simulated)</small>}
                    </strong>
                  </div>
                ) : (
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    {chargingAdvice.detail}
                  </span>
                )}
                <button
                  type="button"
                  className="secondary-button"
                  onClick={onNavigateToCharging}
                  style={{ marginTop: '8px', width: '100%', fontSize: '12px', padding: '6px 10px' }}
                >
                  Inspect Mapped Chargers <ArrowRight size={13} />
                </button>
              </>
            ) : chargingAdvice.status === 'low_margin' ? (
              <>
                <p style={{ margin: '0 0 6px', color: 'var(--accent-amber)', fontWeight: 600 }}>
                  {chargingAdvice.headline}
                </p>
                <button
                  type="button"
                  className="text-button"
                  onClick={onNavigateToCharging}
                  style={{ fontSize: '12px', padding: 0 }}
                >
                  View chargers along route <ArrowRight size={12} />
                </button>
              </>
            ) : (
              <>
                <p style={{ margin: '0 0 6px', color: 'var(--accent-emerald)', fontWeight: 600 }}>
                  {chargingAdvice.headline}
                </p>
                <span style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                  {chargingAdvice.detail}
                </span>
              </>
            )}
          </div>
        </div>

      </div>

      {/* Disclaimers & Notes */}
      <div style={{ marginTop: '14px', paddingTop: '10px', borderTop: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)', lineHeight: '1.4' }}>
          * Results are physics-based planning estimates derived from entered battery capacity ({capacity} kWh), starting charge ({soc}%), health ({route.weatherAdjusted ? `weather factor ${fmt(route.weatherFactor, 3)}` : 'unadjusted'}), and consumption ({consumption} Wh/km). Completion is not guaranteed.
        </p>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          OSRM Route: {fmt(distanceKm, 1)} km · {formatRouteDuration(route.durationSeconds)}
        </span>
      </div>
    </section>
  );
}

export default TripSummary;
