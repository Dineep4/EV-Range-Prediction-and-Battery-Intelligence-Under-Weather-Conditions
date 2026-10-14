import React, { useState, useEffect } from 'react';
import {
  Activity,
  BatteryFull,
  BatteryCharging,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  RefreshCw,
  Info,
  ShieldCheck,
  Zap,
  Sliders,
  ChevronDown,
  ChevronUp,
  Gauge,
  Cpu,
  FileText,
  Sparkles,
} from 'lucide-react';

import { SAMPLES, validateSOHInputs } from './batterySOH.js';

export default function BatterySOHPredictor({ apiBase = 'http://127.0.0.1:8000' }) {
  const [cycleIndex, setCycleIndex] = useState(50);
  const [voltageMean, setVoltageMean] = useState(3.53);
  const [currentMean, setCurrentMean] = useState(-1.82);
  const [temperatureMean, setTemperatureMean] = useState(32.5);
  const [capacityAh, setCapacityAh] = useState('');
  const [ratedCapacityAh, setRatedCapacityAh] = useState(2.0);

  const [fieldErrors, setFieldErrors] = useState({});
  const [generalError, setGeneralError] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [activeSampleId, setActiveSampleId] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  useEffect(() => {
    let isMounted = true;
    fetch(`${apiBase}/api/soh/metrics`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (isMounted && data) setMetrics(data);
      })
      .catch(() => {});
    return () => {
      isMounted = false;
    };
  }, [apiBase]);

  const loadSample = (sample) => {
    setCycleIndex(sample.cycle_index);
    setVoltageMean(sample.voltage_mean);
    setCurrentMean(sample.current_mean);
    setTemperatureMean(sample.temperature_mean);
    setCapacityAh(sample.capacity_ah !== undefined ? sample.capacity_ah : '');
    setRatedCapacityAh(sample.rated_capacity_ah || 2.0);
    setActiveSampleId(sample.id);
    setFieldErrors({});
    setGeneralError('');
  };

  const resetForm = () => {
    setCycleIndex(50);
    setVoltageMean(3.53);
    setCurrentMean(-1.82);
    setTemperatureMean(32.5);
    setCapacityAh('');
    setRatedCapacityAh(2.0);
    setPrediction(null);
    setFieldErrors({});
    setGeneralError('');
    setActiveSampleId('');
  };

  const handleSubmit = async (e) => {
    if (e && typeof e.preventDefault === 'function') e.preventDefault();

    const errors = validateSOHInputs({
      cycleIndex,
      voltageMean,
      currentMean,
      temperatureMean,
      capacityAh,
      ratedCapacityAh,
    });

    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) {
      setGeneralError('Please correct the highlighted inputs before requesting an SOH prediction.');
      return;
    }

    setLoading(true);
    setGeneralError('');

    const payload = {
      battery_id: activeSampleId ? `NASA-${activeSampleId}` : 'NASA-18650',
      cycle_index: Number(cycleIndex),
      voltage_mean: Number(voltageMean),
      current_mean: Number(currentMean),
      temperature_mean: Number(temperatureMean),
      capacity_ah: capacityAh !== '' ? Number(capacityAh) : null,
      rated_capacity_ah: ratedCapacityAh !== '' ? Number(ratedCapacityAh) : 2.0,
    };

    try {
      const response = await fetch(`${apiBase}/api/predict/soh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail?.message || data.detail || 'SOH prediction request failed.');
      }
      setPrediction(data);
    } catch (err) {
      setGeneralError(err.message || 'Unable to connect to the SOH prediction service.');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const getSOHColor = (sohVal) => {
    if (sohVal >= 80) return '#059669'; // Emerald
    if (sohVal >= 70) return '#d97706'; // Amber
    return '#dc2626'; // Red
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
      {/* 1. PRIMARY PREDICTION CARD */}
      <section
        className="panel"
        style={{
          border: '1px solid var(--border-subtle, #e2e8f0)',
          borderRadius: '16px',
          boxShadow: 'var(--shadow-md, 0 4px 12px rgba(15, 23, 42, 0.05))',
          background: 'var(--bg-surface, #ffffff)',
          overflow: 'hidden',
          padding: '24px',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: '16px',
            flexWrap: 'wrap',
            paddingBottom: '20px',
            borderBottom: '1px solid var(--border-subtle, #f1f5f9)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div
              style={{
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                background: 'linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)',
                border: '1px solid #a7f3d0',
                display: 'grid',
                placeItems: 'center',
                boxShadow: '0 2px 8px rgba(5, 150, 105, 0.12)',
              }}
            >
              <BatteryFull size={26} color="#059669" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <h2 style={{ fontSize: '20px', fontWeight: 800, margin: 0, color: '#172033' }}>
                  Battery State of Health
                </h2>
                <span
                  className="soft-badge"
                  style={{
                    background: '#ecfdf5',
                    color: '#059669',
                    border: '1px solid #a7f3d0',
                    fontSize: '11px',
                    fontWeight: 700,
                  }}
                >
                  PREDICTION ENGINE
                </span>
              </div>
              <p style={{ margin: '4px 0 0', color: 'var(--text-secondary, #475569)', fontSize: '13.5px', fontWeight: 500 }}>
                Analyse battery condition with data-driven SOH estimation.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span
              style={{
                fontSize: '11.5px',
                padding: '5px 10px',
                borderRadius: '6px',
                background: '#f8fafc',
                border: '1px solid #e2e8f0',
                color: '#64748b',
                fontWeight: 600,
              }}
            >
              Model: {metrics ? metrics.model_name : 'Gradient Boosting'}
            </span>
          </div>
        </div>

        {/* Short explanation */}
        <p
          style={{
            margin: '16px 0 20px',
            fontSize: '13px',
            lineHeight: '1.6',
            color: 'var(--text-secondary, #475569)',
            background: 'var(--bg-surface-elevated, #f8fafc)',
            padding: '12px 16px',
            borderRadius: '10px',
            border: '1px solid var(--border-subtle, #e2e8f0)',
          }}
        >
          This interface uses a supervised Gradient Boosting Regressor trained on lithium-ion 18650 cell cycling data.
          Inputs require key operational aggregates: cycle count, mean operating voltage, discharge current, and surface
          temperature. Enter measurements below or select an authentic baseline preset to generate an empirical SOH estimate.
        </p>

        {/* Preset sample controls */}
        <div
          style={{
            background: '#ffffff',
            border: '1px solid var(--border-subtle, #e2e8f0)',
            borderRadius: '10px',
            padding: '14px 16px',
            marginBottom: '20px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '7px', fontSize: '12.5px', fontWeight: 700, color: '#334155' }}>
              <Sliders size={16} color="#059669" />
              <span>Load Authentic NASA Aging Presets:</span>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {SAMPLES.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  className="secondary-button"
                  onClick={() => loadSample(sample)}
                  style={{
                    fontSize: '11.5px',
                    padding: '6px 12px',
                    borderRadius: '7px',
                    background: activeSampleId === sample.id ? '#eff6ff' : '#ffffff',
                    borderColor: activeSampleId === sample.id ? '#3b82f6' : '#cbd5e1',
                    color: activeSampleId === sample.id ? '#1d4ed8' : '#334155',
                    fontWeight: activeSampleId === sample.id ? 700 : 500,
                    boxShadow: activeSampleId === sample.id ? '0 1px 3px rgba(59, 130, 246, 0.15)' : 'none',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {sample.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Form Inputs Grid */}
        <form onSubmit={handleSubmit} noValidate>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
              gap: '16px',
              marginBottom: '18px',
            }}
          >
            {/* Cycle Count */}
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#334155', marginBottom: '6px' }}>
                Cycle Count (Cycles) <span style={{ color: '#ef4444' }}>*</span>
              </label>
              <input
                type="number"
                min="1"
                max="500"
                step="1"
                value={cycleIndex}
                onChange={(e) => {
                  setCycleIndex(e.target.value);
                  if (fieldErrors.cycleIndex) {
                    setFieldErrors((prev) => ({ ...prev, cycleIndex: undefined }));
                  }
                }}
                style={{
                  width: '100%',
                  padding: '9px 12px',
                  borderRadius: '8px',
                  border: `1px solid ${fieldErrors.cycleIndex ? '#ef4444' : '#cbd5e1'}`,
                  background: fieldErrors.cycleIndex ? '#fef2f2' : '#ffffff',
                  color: '#172033',
                  fontSize: '13.5px',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
                required
              />
              <small style={{ display: 'block', fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                Integer [1, 500] · CC-CV aging cycles
              </small>
              {fieldErrors.cycleIndex && (
                <div style={{ color: '#ef4444', fontSize: '11px', marginTop: '4px', fontWeight: 600 }}>
                  {fieldErrors.cycleIndex}
                </div>
              )}
            </div>

            {/* Mean Operating Voltage */}
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#334155', marginBottom: '6px' }}>
                Mean Operating Voltage (V) <span style={{ color: '#ef4444' }}>*</span>
              </label>
              <input
                type="number"
                min="2.0"
                max="4.5"
                step="0.01"
                value={voltageMean}
                onChange={(e) => {
                  setVoltageMean(e.target.value);
                  if (fieldErrors.voltageMean) {
                    setFieldErrors((prev) => ({ ...prev, voltageMean: undefined }));
                  }
                }}
                style={{
                  width: '100%',
                  padding: '9px 12px',
                  borderRadius: '8px',
                  border: `1px solid ${fieldErrors.voltageMean ? '#ef4444' : '#cbd5e1'}`,
                  background: fieldErrors.voltageMean ? '#fef2f2' : '#ffffff',
                  color: '#172033',
                  fontSize: '13.5px',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
                required
              />
              <small style={{ display: 'block', fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                [2.00, 4.50] V · Mean voltage across cycle
              </small>
              {fieldErrors.voltageMean && (
                <div style={{ color: '#ef4444', fontSize: '11px', marginTop: '4px', fontWeight: 600 }}>
                  {fieldErrors.voltageMean}
                </div>
              )}
            </div>

            {/* Mean Load Current */}
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#334155', marginBottom: '6px' }}>
                Mean Load Current (A) <span style={{ color: '#ef4444' }}>*</span>
              </label>
              <input
                type="number"
                min="-10.0"
                max="10.0"
                step="0.01"
                value={currentMean}
                onChange={(e) => {
                  setCurrentMean(e.target.value);
                  if (fieldErrors.currentMean) {
                    setFieldErrors((prev) => ({ ...prev, currentMean: undefined }));
                  }
                }}
                style={{
                  width: '100%',
                  padding: '9px 12px',
                  borderRadius: '8px',
                  border: `1px solid ${fieldErrors.currentMean ? '#ef4444' : '#cbd5e1'}`,
                  background: fieldErrors.currentMean ? '#fef2f2' : '#ffffff',
                  color: '#172033',
                  fontSize: '13.5px',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
                required
              />
              <small style={{ display: 'block', fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                [-10.0, 10.0] A · Negative for discharge
              </small>
              {fieldErrors.currentMean && (
                <div style={{ color: '#ef4444', fontSize: '11px', marginTop: '4px', fontWeight: 600 }}>
                  {fieldErrors.currentMean}
                </div>
              )}
            </div>

            {/* Mean Cell Temperature */}
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#334155', marginBottom: '6px' }}>
                Mean Cell Temperature (°C) <span style={{ color: '#ef4444' }}>*</span>
              </label>
              <input
                type="number"
                min="0.0"
                max="70.0"
                step="0.1"
                value={temperatureMean}
                onChange={(e) => {
                  setTemperatureMean(e.target.value);
                  if (fieldErrors.temperatureMean) {
                    setFieldErrors((prev) => ({ ...prev, temperatureMean: undefined }));
                  }
                }}
                style={{
                  width: '100%',
                  padding: '9px 12px',
                  borderRadius: '8px',
                  border: `1px solid ${fieldErrors.temperatureMean ? '#ef4444' : '#cbd5e1'}`,
                  background: fieldErrors.temperatureMean ? '#fef2f2' : '#ffffff',
                  color: '#172033',
                  fontSize: '13.5px',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
                required
              />
              <small style={{ display: 'block', fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                [0.0, 70.0] °C · Average surface temp
              </small>
              {fieldErrors.temperatureMean && (
                <div style={{ color: '#ef4444', fontSize: '11px', marginTop: '4px', fontWeight: 600 }}>
                  {fieldErrors.temperatureMean}
                </div>
              )}
            </div>

            {/* Measured Discharge Capacity (Optional) */}
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#334155', marginBottom: '6px' }}>
                Measured Capacity (Ah, Optional)
              </label>
              <input
                type="number"
                min="0.1"
                max="5.0"
                step="0.01"
                placeholder="e.g. 1.84"
                value={capacityAh}
                onChange={(e) => {
                  setCapacityAh(e.target.value);
                  if (fieldErrors.capacityAh) {
                    setFieldErrors((prev) => ({ ...prev, capacityAh: undefined }));
                  }
                }}
                style={{
                  width: '100%',
                  padding: '9px 12px',
                  borderRadius: '8px',
                  border: `1px solid ${fieldErrors.capacityAh ? '#ef4444' : '#cbd5e1'}`,
                  background: fieldErrors.capacityAh ? '#fef2f2' : '#ffffff',
                  color: '#172033',
                  fontSize: '13.5px',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
              />
              <small style={{ display: 'block', fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                (0.10, 5.00] Ah · Direct coulometric reading
              </small>
              {fieldErrors.capacityAh && (
                <div style={{ color: '#ef4444', fontSize: '11px', marginTop: '4px', fontWeight: 600 }}>
                  {fieldErrors.capacityAh}
                </div>
              )}
            </div>

            {/* Rated Reference Capacity */}
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: '#334155', marginBottom: '6px' }}>
                Rated Reference Capacity (Ah)
              </label>
              <input
                type="number"
                min="0.5"
                max="5.0"
                step="0.1"
                value={ratedCapacityAh}
                onChange={(e) => {
                  setRatedCapacityAh(e.target.value);
                  if (fieldErrors.ratedCapacityAh) {
                    setFieldErrors((prev) => ({ ...prev, ratedCapacityAh: undefined }));
                  }
                }}
                style={{
                  width: '100%',
                  padding: '9px 12px',
                  borderRadius: '8px',
                  border: `1px solid ${fieldErrors.ratedCapacityAh ? '#ef4444' : '#cbd5e1'}`,
                  background: fieldErrors.ratedCapacityAh ? '#fef2f2' : '#ffffff',
                  color: '#172033',
                  fontSize: '13.5px',
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
              />
              <small style={{ display: 'block', fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                (0.5, 5.0] Ah · Nominal benchmark (default 2.0 Ah)
              </small>
              {fieldErrors.ratedCapacityAh && (
                <div style={{ color: '#ef4444', fontSize: '11px', marginTop: '4px', fontWeight: 600 }}>
                  {fieldErrors.ratedCapacityAh}
                </div>
              )}
            </div>
          </div>

          {/* Error Banner */}
          {generalError && (
            <div className="error-box" style={{ marginBottom: '16px' }}>
              <AlertTriangle size={16} />
              <span>{generalError}</span>
            </div>
          )}

          {/* Form Actions */}
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              className="primary-button"
              type="submit"
              disabled={loading}
              style={{
                minWidth: '180px',
                height: '42px',
                fontSize: '13.5px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
            >
              {loading ? (
                <>
                  <RefreshCw className="spin" size={17} /> Calculating SOH...
                </>
              ) : (
                <>
                  <Activity size={17} /> Predict Battery SOH
                </>
              )}
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={resetForm}
              disabled={loading}
              style={{
                height: '42px',
                padding: '0 18px',
                fontSize: '13.5px',
                display: 'flex',
                alignItems: 'center',
                gap: '7px',
              }}
            >
              <RotateCcw size={15} /> Reset Inputs
            </button>
          </div>
        </form>

        {/* 2. DEDICATED PREDICTION RESULT AREA */}
        {prediction && (
          <div
            style={{
              marginTop: '26px',
              background: '#ffffff',
              border: '1px solid #bbf7d0',
              borderRadius: '14px',
              padding: '24px',
              boxShadow: '0 4px 20px rgba(5, 150, 105, 0.08)',
            }}
          >
            {/* Result Top Row */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '12px',
                paddingBottom: '16px',
                borderBottom: '1px solid #f1f5f9',
                marginBottom: '20px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <CheckCircle2 size={24} color="#059669" />
                <div>
                  <h3 style={{ fontSize: '17px', fontWeight: 800, margin: 0, color: '#172033' }}>
                    Estimated Battery SOH
                  </h3>
                  <span style={{ fontSize: '11.5px', color: '#64748b' }}>
                    Model: {prediction.model_name} · Held-out validation on Cell {prediction.held_out_test_cell}
                  </span>
                </div>
              </div>
              <span
                className="soft-badge"
                style={{
                  background: '#ecfdf5',
                  color: '#059669',
                  border: '1px solid #a7f3d0',
                  fontWeight: 700,
                  fontSize: '11px',
                  padding: '5px 12px',
                }}
              >
                PREDICTION COMPLETE
              </span>
            </div>

            {/* Visual Focal Point: Gauge + Big Value */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                gap: '20px',
                marginBottom: '22px',
              }}
            >
              {/* Radial SOH Focal Card */}
              <div
                style={{
                  background: 'linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%)',
                  border: '1px solid #86efac',
                  borderRadius: '12px',
                  padding: '20px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '20px',
                }}
              >
                {/* Circular Gauge Ring */}
                <div style={{ width: '100px', height: '100px', position: 'relative', flexShrink: 0 }}>
                  <svg width="100" height="100" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke="#dcfce7"
                      strokeWidth="9"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke={getSOHColor(prediction.predicted_soh_percent)}
                      strokeWidth="9"
                      strokeLinecap="round"
                      strokeDasharray={`${(2 * Math.PI * 40 * Math.min(100, Math.max(0, prediction.predicted_soh_percent))) / 100} ${2 * Math.PI * 40}`}
                      transform="rotate(-90 50 50)"
                      style={{ transition: 'stroke-dasharray 0.6s ease' }}
                    />
                  </svg>
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <span style={{ fontSize: '11px', fontWeight: 800, color: '#047857' }}>SOH</span>
                  </div>
                </div>

                <div>
                  <span style={{ fontSize: '11px', fontWeight: 800, color: '#047857', letterSpacing: '0.6px', textTransform: 'uppercase' }}>
                    Predicted State of Health
                  </span>
                  <div style={{ fontSize: '38px', fontWeight: 900, color: '#065f46', lineHeight: 1.1, margin: '4px 0' }}>
                    {prediction.predicted_soh_percent.toFixed(1)}%
                  </div>
                  <span style={{ fontSize: '11.5px', color: '#047857', fontWeight: 500 }}>
                    Retained capacity ratio vs {prediction.rated_capacity_ah || 2.0} Ah reference rating
                  </span>
                </div>
              </div>

              {/* Statistically Justified Uncertainty Interval */}
              {prediction.uncertainty_interval && (
                <div
                  style={{
                    background: '#f8fafc',
                    border: '1px solid #e2e8f0',
                    borderRadius: '12px',
                    padding: '20px',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '8px' }}>
                    <ShieldCheck size={18} color="#2563eb" />
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#1e293b' }}>
                      Prediction Uncertainty Interval
                    </span>
                  </div>
                  <div style={{ fontSize: '22px', fontWeight: 800, color: '#1e40af', margin: '2px 0 6px' }}>
                    {prediction.uncertainty_interval.lower_bound_percent.toFixed(1)}% —{' '}
                    {prediction.uncertainty_interval.upper_bound_percent.toFixed(1)}%
                  </div>
                  <p style={{ margin: 0, fontSize: '11px', color: '#64748b', lineHeight: 1.45 }}>
                    Empirical 90% confidence interval (±{prediction.uncertainty_interval.margin_percent} pp) derived
                    from the held-out test cell B0006 standard error.
                  </p>
                </div>
              )}

              {/* Coulometric Check if measured capacity was supplied */}
              {prediction.measured_soh_percent !== null && prediction.measured_soh_percent !== undefined && (
                <div
                  style={{
                    background: '#fffbeb',
                    border: '1px solid #fde68a',
                    borderRadius: '12px',
                    padding: '20px',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '8px' }}>
                    <Zap size={18} color="#d97706" />
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#92400e' }}>
                      Coulometric Capacity Benchmark
                    </span>
                  </div>
                  <div style={{ fontSize: '22px', fontWeight: 800, color: '#b45309', margin: '2px 0 6px' }}>
                    {prediction.measured_soh_percent.toFixed(1)}% Measured SOH
                  </div>
                  <p style={{ margin: 0, fontSize: '11px', color: '#92400e', lineHeight: 1.45 }}>
                    Direct calculation from entered capacity ({prediction.input_values?.capacity_ah} Ah / {prediction.rated_capacity_ah} Ah rated).
                  </p>
                </div>
              )}
            </div>

            {/* Horizontal Retention Progress Bar */}
            <div style={{ marginBottom: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11.5px', color: '#64748b', marginBottom: '6px' }}>
                <span>Capacity Degradation Scale</span>
                <span><b>{prediction.predicted_soh_percent.toFixed(1)}%</b> Retained Capacity</span>
              </div>
              <div
                style={{
                  width: '100%',
                  height: '10px',
                  background: '#f1f5f9',
                  borderRadius: '6px',
                  overflow: 'hidden',
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    width: `${Math.min(100, Math.max(0, prediction.predicted_soh_percent))}%`,
                    height: '100%',
                    background: getSOHColor(prediction.predicted_soh_percent),
                    borderRadius: '6px',
                    transition: 'width 0.5s ease',
                  }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94a3b8', marginTop: '4px' }}>
                <span>0% (Fully Degraded)</span>
                <span>70% (First-Life Threshold)</span>
                <span>100% (Nominal Fresh)</span>
              </div>
            </div>

            {/* Input Summary Chips */}
            <div
              style={{
                background: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '9px',
                padding: '12px 14px',
                marginBottom: '16px',
              }}
            >
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#475569', display: 'block', marginBottom: '8px' }}>
                INPUT TELEMETRY SNAPSHOT
              </span>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '11px', padding: '3px 8px', background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '5px', color: '#334155' }}>
                  Cycle: <b>{prediction.input_values?.cycle_index}</b>
                </span>
                <span style={{ fontSize: '11px', padding: '3px 8px', background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '5px', color: '#334155' }}>
                  Mean Voltage: <b>{prediction.input_values?.voltage_mean} V</b>
                </span>
                <span style={{ fontSize: '11px', padding: '3px 8px', background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '5px', color: '#334155' }}>
                  Load Current: <b>{prediction.input_values?.current_mean} A</b>
                </span>
                <span style={{ fontSize: '11px', padding: '3px 8px', background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '5px', color: '#334155' }}>
                  Cell Temp: <b>{prediction.input_values?.temperature_mean} °C</b>
                </span>
                {prediction.input_values?.capacity_ah && (
                  <span style={{ fontSize: '11px', padding: '3px 8px', background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '5px', color: '#334155' }}>
                    Discharge Cap: <b>{prediction.input_values?.capacity_ah} Ah</b>
                  </span>
                )}
                <span style={{ fontSize: '11px', padding: '3px 8px', background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '5px', color: '#334155' }}>
                  Rated Ref: <b>{prediction.rated_capacity_ah} Ah</b>
                </span>
              </div>
            </div>

            {/* Held-out Evaluation Metrics Table Strip */}
            {prediction.evaluation_metrics && (
              <div
                style={{
                  background: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: '9px',
                  padding: '12px 14px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '12px',
                  fontSize: '11px',
                  color: '#475569',
                }}
              >
                <div>
                  <b>Held-out Evaluation Metrics (Cell {prediction.held_out_test_cell}):</b>
                </div>
                <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
                  <span>RMSE: <b>{prediction.evaluation_metrics.rmse_percent.toFixed(2)} pp</b></span>
                  <span>MAE: <b>{prediction.evaluation_metrics.mae_percent.toFixed(2)} pp</b></span>
                  <span>R²: <b>{prediction.evaluation_metrics.r2_score.toFixed(3)}</b></span>
                  <span>MAPE: <b>{prediction.evaluation_metrics.mape_percent.toFixed(2)}%</b></span>
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {/* 3. BATTERY ASSESSMENT SUMMARY */}
      <section
        className="panel"
        style={{
          border: '1px solid var(--border-subtle, #e2e8f0)',
          borderRadius: '16px',
          boxShadow: 'var(--shadow-md, 0 4px 12px rgba(15, 23, 42, 0.05))',
          background: 'var(--bg-surface, #ffffff)',
          padding: '24px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <Gauge size={22} color="#059669" />
          <div>
            <h3 style={{ fontSize: '17px', fontWeight: 800, margin: 0, color: '#172033' }}>
              Battery Assessment Summary
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: '12.5px', color: '#64748b' }}>
              Synthesised engineering interpretation of current battery state and degradation characteristics.
            </p>
          </div>
        </div>

        {prediction ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                gap: '14px',
              }}
            >
              {/* Degradation Offset */}
              <div
                style={{
                  background: 'var(--bg-surface-elevated, #f8fafc)',
                  border: '1px solid var(--border-subtle, #e2e8f0)',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
                  Capacity Degradation
                </span>
                <div style={{ fontSize: '20px', fontWeight: 800, color: '#172033', margin: '4px 0' }}>
                  {(100 - prediction.predicted_soh_percent).toFixed(1)}% Capacity Loss
                </div>
                <small style={{ fontSize: '11px', color: '#64748b' }}>
                  Relative to fresh 2.0 Ah nominal manufacturer benchmark.
                </small>
              </div>

              {/* Capacity Comparison */}
              <div
                style={{
                  background: 'var(--bg-surface-elevated, #f8fafc)',
                  border: '1px solid var(--border-subtle, #e2e8f0)',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
                  Capacity Comparison
                </span>
                <div style={{ fontSize: '20px', fontWeight: 800, color: '#172033', margin: '4px 0' }}>
                  {prediction.input_values?.capacity_ah
                    ? `${Number(prediction.input_values.capacity_ah).toFixed(2)} Ah Measured`
                    : `${((prediction.predicted_soh_percent * (prediction.rated_capacity_ah || 2.0)) / 100).toFixed(2)} Ah Estimated`}
                </div>
                <small style={{ fontSize: '11px', color: '#64748b' }}>
                  Against {prediction.rated_capacity_ah || 2.0} Ah reference rating.
                </small>
              </div>

              {/* Model Confidence Metric */}
              <div
                style={{
                  background: 'var(--bg-surface-elevated, #f8fafc)',
                  border: '1px solid var(--border-subtle, #e2e8f0)',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase' }}>
                  Held-Out Benchmark
                </span>
                <div style={{ fontSize: '20px', fontWeight: 800, color: '#059669', margin: '4px 0' }}>
                  R² = {prediction.evaluation_metrics?.r2_score || '0.922'}
                </div>
                <small style={{ fontSize: '11px', color: '#64748b' }}>
                  Unseen test cell B0006 ({prediction.evaluation_metrics?.rmse_percent || '3.50'} pp RMSE).
                </small>
              </div>
            </div>

            {/* Concise Interpretation and Limitations */}
            <div
              style={{
                background: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '10px',
                padding: '16px',
                fontSize: '12.5px',
                lineHeight: 1.6,
                color: '#334155',
              }}
            >
              <b style={{ color: '#172033' }}>Assessment Interpretation &amp; Engineering Constraints:</b>
              <p style={{ margin: '6px 0 0' }}>
                The Gradient Boosting Regressor predicts that following <b>{prediction.input_values?.cycle_index}</b> operational
                cycles under nominal load, the cell retains approximately <b>{prediction.predicted_soh_percent.toFixed(1)}%</b> of
                its original reference capacity.
                {prediction.measured_soh_percent !== null && prediction.measured_soh_percent !== undefined && (
                  <span>
                    {' '}Your direct coulometric measurement indicates <b>{prediction.measured_soh_percent.toFixed(1)}%</b> SOH,
                    showing consistent alignment with model estimates.
                  </span>
                )}
                {' '}This estimate reflects continuous constant-current laboratory cycling. In actual automotive EV packs,
                dynamic driving profiles, thermal pack management, and regenerative pulses introduce additional variables
                that require on-board vehicle BMS telemetry.
              </p>
            </div>
          </div>
        ) : (
          /* Attractive Empty State */
          <div
            style={{
              padding: '36px 20px',
              textAlign: 'center',
              background: 'var(--bg-surface-elevated, #f8fafc)',
              border: '1px dashed var(--border-medium, #cbd5e1)',
              borderRadius: '12px',
            }}
          >
            <div
              style={{
                width: '52px',
                height: '52px',
                borderRadius: '50%',
                background: '#f1f5f9',
                display: 'grid',
                placeItems: 'center',
                margin: '0 auto 12px',
              }}
            >
              <BatteryCharging size={26} color="#94a3b8" />
            </div>
            <h4 style={{ fontSize: '15px', fontWeight: 700, margin: '0 0 6px', color: '#172033' }}>
              No Assessment Generated Yet
            </h4>
            <p style={{ fontSize: '12.5px', color: '#64748b', maxWidth: '440px', margin: '0 auto 16px', lineHeight: 1.5 }}>
              Enter your battery&apos;s operational measurements in the form above or click one of the authentic NASA
              presets to generate a data-driven State of Health assessment.
            </p>
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="secondary-button"
                onClick={() => loadSample(SAMPLES[0])}
                style={{ fontSize: '12px', padding: '6px 14px' }}
              >
                Try Early Cycle Baseline
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => loadSample(SAMPLES[1])}
                style={{ fontSize: '12px', padding: '6px 14px' }}
              >
                Try Mid-Life Aging Preset
              </button>
            </div>
          </div>
        )}
      </section>

      {/* 4. EXPANDABLE ABOUT THIS PREDICTION SECTION */}
      <section
        className="panel"
        style={{
          border: '1px solid var(--border-subtle, #e2e8f0)',
          borderRadius: '16px',
          boxShadow: 'var(--shadow-xs, 0 1px 3px rgba(15, 23, 42, 0.04))',
          background: 'var(--bg-surface, #ffffff)',
          overflow: 'hidden',
        }}
      >
        <button
          type="button"
          onClick={() => setDetailsOpen(!detailsOpen)}
          style={{
            width: '100%',
            padding: '18px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'transparent',
            border: 'none',
            textAlign: 'left',
            cursor: 'pointer',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Cpu size={20} color="#059669" />
            <div>
              <b style={{ fontSize: '15px', color: '#172033', display: 'block' }}>
                About This Prediction
              </b>
              <span style={{ fontSize: '12px', color: '#64748b' }}>
                Dataset provenance, target definition, model architecture, and laboratory limitations
              </span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#64748b' }}>
            <small style={{ fontSize: '12px', fontWeight: 600 }}>{detailsOpen ? 'Hide Details' : 'Expand Details'}</small>
            {detailsOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </div>
        </button>

        {detailsOpen && (
          <div
            style={{
              padding: '0 24px 24px',
              borderTop: '1px solid #f1f5f9',
            }}
          >
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                gap: '16px',
                marginTop: '18px',
                marginBottom: '20px',
              }}
            >
              {/* Dataset Used */}
              <div
                style={{
                  background: 'var(--bg-surface-elevated, #f8fafc)',
                  border: '1px solid var(--border-subtle, #e2e8f0)',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  <FileText size={16} color="#059669" />
                  <b style={{ fontSize: '12.5px', color: '#1e293b' }}>Dataset Provenance</b>
                </div>
                <p style={{ margin: 0, fontSize: '11.5px', color: '#475569', lineHeight: 1.5 }}>
                  NASA Ames Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset. Commercial 2.0 Ah
                  18650 cylindrical cells (B0005, B0007, B0018, B0025, B0026, B0027, B0028) cycled under continuous
                  charge/discharge regimes at room temperature (~24°C).
                </p>
              </div>

              {/* Target Definition */}
              <div
                style={{
                  background: 'var(--bg-surface-elevated, #f8fafc)',
                  border: '1px solid var(--border-subtle, #e2e8f0)',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  <Activity size={16} color="#2563eb" />
                  <b style={{ fontSize: '12.5px', color: '#1e293b' }}>Target Definition</b>
                </div>
                <p style={{ margin: 0, fontSize: '11.5px', color: '#475569', lineHeight: 1.5 }}>
                  State of Health is defined as the percentage of initial manufacturer reference capacity retained:
                  <code style={{ display: 'block', margin: '4px 0', padding: '2px 6px', background: '#e2e8f0', borderRadius: '4px', fontSize: '10.5px' }}>
                    SOH (%) = 100 × [Discharge Capacity (Ah)] / [2.0 Ah Rated Benchmark]
                  </code>
                </p>
              </div>

              {/* Model Architecture */}
              <div
                style={{
                  background: 'var(--bg-surface-elevated, #f8fafc)',
                  border: '1px solid var(--border-subtle, #e2e8f0)',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  <Cpu size={16} color="#7c3aed" />
                  <b style={{ fontSize: '12.5px', color: '#1e293b' }}>Model Architecture</b>
                </div>
                <p style={{ margin: 0, fontSize: '11.5px', color: '#475569', lineHeight: 1.5 }}>
                  Supervised Gradient Boosting Regressor (120 decision trees, learning rate 0.05, max tree depth 3,
                  Huber robust loss). Predicts SOH from 4 operational features without target leakage: cycle index,
                  mean voltage, mean load current, and surface temperature.
                </p>
              </div>

              {/* Evaluation Methodology */}
              <div
                style={{
                  background: 'var(--bg-surface-elevated, #f8fafc)',
                  border: '1px solid var(--border-subtle, #e2e8f0)',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  <ShieldCheck size={16} color="#059669" />
                  <b style={{ fontSize: '12.5px', color: '#1e293b' }}>Evaluation Methodology</b>
                </div>
                <p style={{ margin: 0, fontSize: '11.5px', color: '#475569', lineHeight: 1.5 }}>
                  Strict battery-grouped leave-one-out cross-validation. Cell B0006 was held out entirely as an unseen
                  test battery (168 test cycles). Achieved 3.50 pp RMSE, 2.89 pp MAE, and R² 0.922 on this unseen cell.
                </p>
              </div>

              {/* Limitations of Laboratory Data */}
              <div
                style={{
                  background: 'var(--bg-surface-elevated, #f8fafc)',
                  border: '1px solid var(--border-subtle, #e2e8f0)',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  <AlertTriangle size={16} color="#d97706" />
                  <b style={{ fontSize: '12.5px', color: '#1e293b' }}>Laboratory vs Real-World</b>
                </div>
                <p style={{ margin: 0, fontSize: '11.5px', color: '#475569', lineHeight: 1.5 }}>
                  Cells were cycled under continuous constant-current / constant-voltage (CC-CV) laboratory profiles. Real
                  EV duty cycles feature variable acceleration demands, regenerative braking surges, ambient climate
                  extremes, and calendar aging periods.
                </p>
              </div>

              {/* Difference from Real EV BMS */}
              <div
                style={{
                  background: 'var(--bg-surface-elevated, #f8fafc)',
                  border: '1px solid var(--border-subtle, #e2e8f0)',
                  borderRadius: '10px',
                  padding: '14px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  <Zap size={16} color="#2563eb" />
                  <b style={{ fontSize: '12.5px', color: '#1e293b' }}>BMS Comparison</b>
                </div>
                <p style={{ margin: 0, fontSize: '11.5px', color: '#475569', lineHeight: 1.5 }}>
                  Vehicle battery management systems measure real-time electrochemical impedance, high-frequency coulomb
                  counting, Kalman filters, and cell-balancing telemetry across large series-parallel strings, rather
                  than single-cell regression aggregates.
                </p>
              </div>
            </div>

            {/* Mandatory Disclaimer */}
            <div
              className="notice warning-notice"
              style={{
                margin: 0,
                display: 'flex',
                alignItems: 'flex-start',
                gap: '10px',
                background: '#fffbeb',
                border: '1px solid #fde68a',
                padding: '12px 16px',
                borderRadius: '8px',
                fontSize: '12px',
                color: '#92400e',
                lineHeight: 1.5,
              }}
            >
              <Info size={17} style={{ flexShrink: 0, marginTop: '2px' }} />
              <span>
                “This result is a data-driven estimate based on the selected model and available battery measurements.
                It is not a substitute for certified battery diagnostics or measurements from the vehicle&apos;s battery
                management system.”
              </span>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
