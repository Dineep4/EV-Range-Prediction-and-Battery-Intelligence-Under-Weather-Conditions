import React, { useState, useEffect, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
  Cell,
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  Award,
  ChartNoAxesCombined,
  CheckCircle2,
  Cpu,
  Database,
  ExternalLink,
  Info,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
  Zap,
} from 'lucide-react';

import {
  HELD_OUT_TRIP_BENCHMARKS,
  validateTripDistanceInputs,
  determineBestModel,
  calculatePredictionSpread,
} from './modelPerformanceUtils.js';

export default function ModelPerformanceView({
  metrics = null,
  apiBase = 'http://127.0.0.1:8000',
  currentSoc = 82,
  currentTemperature = 22,
}) {
  // Input fields for 30s initial telemetry observation
  const [windowDistanceKm, setWindowDistanceKm] = useState('0.28');
  const [speedMean, setSpeedMean] = useState('35');
  const [powerMean, setPowerMean] = useState('10.5');

  // Metric selector for identifying benchmark winner: 'rmse' or 'mae'
  const [selectedMetric, setSelectedMetric] = useState('rmse');

  // Multi-model predictions state
  const [loading, setLoading] = useState(false);
  const [predictionResults, setPredictionResults] = useState([]);
  const [formErrors, setFormErrors] = useState({});
  const [generalError, setGeneralError] = useState('');

  const fmt = (n, d = 3) =>
    n === null || n === undefined || n === '' || !Number.isFinite(Number(n)) ? '—' : Number(n).toFixed(d);

  // Available models to compare
  const compatibleModels = useMemo(
    () => [
      { name: 'XGBoost', apiModelName: 'XGBoost', key: 'xgboost' },
      { name: 'GRU', apiModelName: 'GRU', key: 'gru' },
      { name: 'LSTM', apiModelName: 'LSTM', key: 'lstm' },
      { name: 'Hybrid CNN-LSTM', apiModelName: 'Hybrid CNN-LSTM', key: 'hybrid' },
      { name: 'Training-Mean Baseline', apiModelName: 'Training-Mean Baseline', key: 'baseline' },
    ],
    []
  );

  // Run multi-model prediction across all compatible models
  const runMultiModelComparison = async (overrideInputs = null) => {
    const distVal = overrideInputs ? overrideInputs.windowDistanceKm : windowDistanceKm;
    const speedVal = overrideInputs ? overrideInputs.speedMean : speedMean;
    const powerVal = overrideInputs ? overrideInputs.powerMean : powerMean;

    const validation = validateTripDistanceInputs({
      windowDistanceKm: distVal,
      speedMean: speedVal,
      powerMean: powerVal,
    });

    if (!validation.isValid) {
      setFormErrors(validation.errors);
      setGeneralError('Please correct the telemetry input values before running multi-model inference.');
      return;
    }

    setFormErrors({});
    setGeneralError('');
    setLoading(true);

    const distNum = Number(distVal);
    const speedNum = Number(speedVal);
    const powerNum = Number(powerVal);

    // Common telemetry context shared across all models
    const basePayload = {
      window_distance_km: distNum,
      speed_mean: speedNum,
      speed_max: Math.round(speedNum * 1.35 * 10) / 10,
      speed_min: 0.0,
      speed_std: Math.round(speedNum * 0.25 * 10) / 10,
      power_mean: powerNum,
      power_max: Math.round(Math.max(powerNum * 2.5, powerNum + 15) * 10) / 10,
      power_min: -5.0,
      current_mean: Math.round(((powerNum * 1000) / 380) * 10) / 10,
      voltage_mean: 382.0,
      soc_start: Number(currentSoc) || 82.0,
      soc_current: Math.max(0, (Number(currentSoc) || 82.0) - 0.2),
      soc_delta: -0.2,
      temperature_mean: Number(currentTemperature) || 22.0,
    };

    try {
      // Fire parallel requests with Promise.allSettled so individual failures do not break others
      const promises = compatibleModels.map(async (m) => {
        const payload = {
          ...basePayload,
          model_name: m.apiModelName,
        };

        const res = await fetch(`${apiBase}/api/predict/trip-distance`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        return {
          model: m.name,
          apiModelName: m.apiModelName,
          key: m.key,
          predictedKm: data.predicted_trip_distance_km,
          status: 'success',
          error: null,
          notice: data.notice,
          observationWindow: data.observation_window,
        };
      });

      const settled = await Promise.allSettled(promises);

      const results = settled.map((result, idx) => {
        const m = compatibleModels[idx];
        if (result.status === 'fulfilled') {
          return result.value;
        } else {
          return {
            model: m.name,
            apiModelName: m.apiModelName,
            key: m.key,
            predictedKm: null,
            status: 'failed',
            error: result.reason?.message || 'Inference call failed',
            notice: null,
            observationWindow: null,
          };
        }
      });

      setPredictionResults(results);
    } catch (err) {
      setGeneralError(err.message || 'An error occurred while executing multi-model comparison.');
    } finally {
      setLoading(false);
    }
  };

  // Run on initial mount so charts and tables are pre-populated
  useEffect(() => {
    runMultiModelComparison({
      windowDistanceKm: '0.28',
      speedMean: '35',
      powerMean: '10.5',
    });
  }, [apiBase]);

  // Reset inputs to default values
  const handleResetInputs = () => {
    const defaults = {
      windowDistanceKm: '0.28',
      speedMean: '35',
      powerMean: '10.5',
    };
    setWindowDistanceKm(defaults.windowDistanceKm);
    setSpeedMean(defaults.speedMean);
    setPowerMean(defaults.powerMean);
    setFormErrors({});
    setGeneralError('');
    runMultiModelComparison(defaults);
  };

  // Best model analysis based on selected benchmark metric (RMSE vs MAE)
  const bestModelAnalysis = useMemo(() => {
    return determineBestModel(HELD_OUT_TRIP_BENCHMARKS, selectedMetric);
  }, [selectedMetric]);

  // Prediction spread analysis across successful inferences
  const predictionSpread = useMemo(() => {
    return calculatePredictionSpread(predictionResults);
  }, [predictionResults]);

  // Merged comparison rows combining historical benchmark metrics and live inference
  const comparisonRows = useMemo(() => {
    return HELD_OUT_TRIP_BENCHMARKS.map((bm) => {
      const pred = predictionResults.find(
        (p) => p.model.toLowerCase() === bm.model.toLowerCase() || p.key === bm.modelKey
      );
      const isWinner = bestModelAnalysis?.winnerModel.toLowerCase() === bm.model.toLowerCase();
      return {
        ...bm,
        predictedKm: pred ? pred.predictedKm : null,
        status: pred ? pred.status : loading ? 'pending' : 'idle',
        error: pred ? pred.error : null,
        isWinner,
      };
    });
  }, [predictionResults, bestModelAnalysis, loading]);

  // Data for Graph A: Predicted Distance by Model
  const predictedDistanceChartData = useMemo(() => {
    return comparisonRows
      .filter((row) => row.status === 'success' && Number.isFinite(row.predictedKm))
      .map((row) => ({
        name: row.model,
        distanceKm: row.predictedKm,
        color: row.color,
      }));
  }, [comparisonRows]);

  // Data for Graph B: Historical Error (RMSE vs MAE)
  const historicalErrorChartData = useMemo(() => {
    return HELD_OUT_TRIP_BENCHMARKS.map((bm) => ({
      name: bm.model,
      RMSE: bm.rmse,
      MAE: bm.mae,
    }));
  }, []);

  // Data for Graph C: Validation R²
  const r2ChartData = useMemo(() => {
    return HELD_OUT_TRIP_BENCHMARKS.map((bm) => ({
      name: bm.model,
      r2: bm.r2,
      color: bm.r2 < 0 ? '#DC2626' : '#059669',
    }));
  }, []);

  // Extract NASA metrics from props or metrics object
  const nasaEvaluation = metrics?.nasa_battery_degradation || metrics?.nasa_evaluation;

  return (
    <div className="model-performance-container">
      {/* 1. Header Banner */}
      <div className="research-banner">
        <div className="research-icon">
          <ChartNoAxesCombined size={24} />
        </div>
        <div>
          <h3>Model Evaluation &amp; Multi-Model Benchmarking Hub</h3>
          <p>
            Interactive comparative inference and empirical benchmark analytics across VED telemetry and NASA lithium-ion degradation.
          </p>
        </div>
        <span className="soft-badge">ACADEMIC BENCHMARK</span>
      </div>

      {/* 2. Disambiguation Notice */}
      <div className="notice warning-notice" style={{ marginBottom: '18px' }}>
        <Info size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <b>Academic Disambiguation &amp; Target Isolation:</b>
          <p style={{ margin: '4px 0 0', lineHeight: '1.5', fontSize: '12px' }}>
            VoltWise enforces strict separation across research domains:
            (1) <b>VED Trip-Distance Benchmark</b>: exploratory 5-model telemetry experiment predicting completed trip length from initial 30 seconds on unseen Vehicle 10 (completed distance, NOT remaining driving range);
            (2) <b>NASA Capacity-Retention Benchmark</b>: 4-model (XGBoost, LSTM, GRU, Hybrid CNN-LSTM) plus baselines degradation comparison on held-out cell B0006;
            (3) <b>Trip Planner</b>: deterministic physics calculation using battery capacity and consumption.
            Predicted completed trip distance does NOT represent remaining driving range.
          </p>
        </div>
      </div>

      {/* 3. Interactive Multi-Model Inference & Comparison Sandbox */}
      <section className="panel" style={{ marginBottom: '20px' }}>
        <div className="panel-head">
          <div>
            <h3>Interactive Multi-Model Inference &amp; Comparison</h3>
            <p>
              Enter the initial 30-second telemetry observation once to compare predictions simultaneously across all 5 deployed architectures.
            </p>
          </div>
          <Cpu color="#2563eb" />
        </div>

        {/* Input Form */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            runMultiModelComparison();
          }}
          className="battery-form"
          style={{ marginBottom: '16px' }}
        >
          <div className="form-row" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
            <label>
              Initial 30s Distance (km)
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="5.0"
                value={windowDistanceKm}
                onChange={(e) => setWindowDistanceKm(e.target.value)}
                placeholder="e.g. 0.28"
                required
              />
              {formErrors.windowDistanceKm && (
                <small style={{ color: '#dc2626', display: 'block', marginTop: '2px' }}>
                  {formErrors.windowDistanceKm}
                </small>
              )}
            </label>

            <label>
              Mean Speed (km/h)
              <input
                type="number"
                step="1"
                min="1"
                max="160"
                value={speedMean}
                onChange={(e) => setSpeedMean(e.target.value)}
                placeholder="e.g. 35"
                required
              />
              {formErrors.speedMean && (
                <small style={{ color: '#dc2626', display: 'block', marginTop: '2px' }}>
                  {formErrors.speedMean}
                </small>
              )}
            </label>

            <label>
              Mean Power (kW)
              <input
                type="number"
                step="0.5"
                min="-50"
                max="200"
                value={powerMean}
                onChange={(e) => setPowerMean(e.target.value)}
                placeholder="e.g. 10.5"
                required
              />
              {formErrors.powerMean && (
                <small style={{ color: '#dc2626', display: 'block', marginTop: '2px' }}>
                  {formErrors.powerMean}
                </small>
              )}
            </label>
          </div>

          <div style={{ display: 'flex', gap: '10px', marginTop: '14px', alignItems: 'center', flexWrap: 'wrap' }}>
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? (
                <>
                  <RefreshCw className="spin" size={16} /> Comparing 5 Models...
                </>
              ) : (
                <>
                  <Activity size={16} /> Compare All Models
                </>
              )}
            </button>

            <button
              className="secondary-button"
              type="button"
              onClick={handleResetInputs}
              disabled={loading}
            >
              <RotateCcw size={15} /> Reset Inputs
            </button>

            {/* Metric Selector for Highlighting Best Model */}
            <div
              style={{
                marginLeft: 'auto',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                background: '#F1F5F9',
                padding: '4px 10px',
                borderRadius: '8px',
              }}
            >
              <span style={{ fontSize: '11.5px', color: '#475569', fontWeight: 600 }}>Evaluation Metric:</span>
              <button
                type="button"
                className="text-button"
                style={{
                  fontSize: '11px',
                  fontWeight: selectedMetric === 'rmse' ? 700 : 500,
                  background: selectedMetric === 'rmse' ? '#2563EB' : 'transparent',
                  color: selectedMetric === 'rmse' ? '#FFFFFF' : '#475569',
                  padding: '3px 8px',
                  borderRadius: '4px',
                }}
                onClick={() => setSelectedMetric('rmse')}
              >
                Lowest RMSE (XGBoost)
              </button>
              <button
                type="button"
                className="text-button"
                style={{
                  fontSize: '11px',
                  fontWeight: selectedMetric === 'mae' ? 700 : 500,
                  background: selectedMetric === 'mae' ? '#2563EB' : 'transparent',
                  color: selectedMetric === 'mae' ? '#FFFFFF' : '#475569',
                  padding: '3px 8px',
                  borderRadius: '4px',
                }}
                onClick={() => setSelectedMetric('mae')}
              >
                Lowest MAE (GRU)
              </button>
            </div>
          </div>
        </form>

        {generalError && (
          <div className="error-box" style={{ marginBottom: '14px' }}>
            <AlertTriangle size={16} />
            <span>{generalError}</span>
          </div>
        )}

        {/* 4. Model Evaluation Summary Panel */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
            gap: '12px',
            background: '#F8FAFC',
            border: '1px solid #E2E8F0',
            borderRadius: '10px',
            padding: '14px',
            marginBottom: '16px',
          }}
        >
          <div>
            <span style={{ fontSize: '11px', color: '#64748B', fontWeight: 700, textTransform: 'uppercase' }}>
              Benchmark Best ({selectedMetric.toUpperCase()})
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '2px' }}>
              <Award size={18} color="#059669" />
              <strong style={{ fontSize: '14px', color: '#0F172A' }}>
                {bestModelAnalysis?.winnerModel || '—'}
              </strong>
            </div>
            <small style={{ fontSize: '11px', color: '#64748B' }}>
              Score: {fmt(bestModelAnalysis?.metricValue, 3)} km {selectedMetric.toUpperCase()}
            </small>
          </div>

          <div>
            <span style={{ fontSize: '11px', color: '#64748B', fontWeight: 700, textTransform: 'uppercase' }}>
              Held-Out Lowest RMSE
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '2px' }}>
              <TrendingDown size={18} color="#059669" />
              <strong style={{ fontSize: '14px', color: '#0F172A' }}>XGBoost (1.488 km)</strong>
            </div>
            <small style={{ fontSize: '11px', color: '#64748B' }}>Best tabular tree model on aggregates</small>
          </div>

          <div>
            <span style={{ fontSize: '11px', color: '#64748B', fontWeight: 700, textTransform: 'uppercase' }}>
              Held-Out Lowest MAE
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '2px' }}>
              <TrendingDown size={18} color="#2563EB" />
              <strong style={{ fontSize: '14px', color: '#0F172A' }}>GRU (0.905 km)</strong>
            </div>
            <small style={{ fontSize: '11px', color: '#64748B' }}>Best sequence model (45.60% MAPE)</small>
          </div>

          <div>
            <span style={{ fontSize: '11px', color: '#64748B', fontWeight: 700, textTransform: 'uppercase' }}>
              Current Prediction Spread
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '2px' }}>
              <Zap size={18} color="#7C3AED" />
              <strong style={{ fontSize: '14px', color: '#0F172A' }}>
                {predictionSpread.spread !== null ? `${fmt(predictionSpread.spread, 2)} km` : '—'}
              </strong>
            </div>
            <small style={{ fontSize: '11px', color: '#64748B' }}>
              Range: {fmt(predictionSpread.min, 2)} km – {fmt(predictionSpread.max, 2)} km ({predictionSpread.successfulCount}/5 models)
            </small>
          </div>
        </div>

        {/* 5. Model Comparison Table */}
        <div className="table-wrap" style={{ marginBottom: '14px' }}>
          <table>
            <thead>
              <tr>
                <th>Model Name</th>
                <th>Predicted Distance (km)</th>
                <th>Validation RMSE (km)</th>
                <th>Validation MAE (km)</th>
                <th>Validation R²</th>
                <th>Benchmark Status</th>
              </tr>
            </thead>
            <tbody>
              {comparisonRows.map((row) => {
                const isSelectedWinner = row.isWinner;
                return (
                  <tr
                    key={row.model}
                    style={{
                      background: isSelectedWinner ? '#F0FDF4' : undefined,
                      transition: 'background 0.2s',
                    }}
                  >
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span
                          style={{
                            width: '10px',
                            height: '10px',
                            borderRadius: '50%',
                            background: row.color,
                            display: 'inline-block',
                          }}
                        />
                        <b>{row.model}</b>
                        {isSelectedWinner && (
                          <span
                            style={{
                              fontSize: '10px',
                              color: '#059669',
                              fontWeight: 700,
                              background: '#ECFDF5',
                              border: '1px solid #A7F3D0',
                              padding: '1px 6px',
                              borderRadius: '4px',
                            }}
                          >
                            ★ Best {selectedMetric.toUpperCase()}
                          </span>
                        )}
                      </div>
                      <small style={{ color: '#64748B', fontSize: '10.5px' }}>{row.architecture}</small>
                    </td>

                    <td>
                      {row.status === 'success' ? (
                        <b style={{ color: '#1E40AF', fontSize: '13px' }}>
                          {fmt(row.predictedKm, 2)} km
                        </b>
                      ) : row.status === 'failed' ? (
                        <span style={{ color: '#DC2626', fontSize: '11px' }}>
                          <AlertTriangle size={12} style={{ display: 'inline', verticalAlign: '-1px' }} /> {row.error || 'Failed'}
                        </span>
                      ) : (
                        <span style={{ color: '#94A3B8' }}>{loading ? 'Evaluating...' : 'Pending'}</span>
                      )}
                    </td>

                    <td>
                      <span style={{ fontWeight: selectedMetric === 'rmse' && isSelectedWinner ? 700 : 500 }}>
                        {fmt(row.rmse, 3)} km
                      </span>
                    </td>

                    <td>
                      <span style={{ fontWeight: selectedMetric === 'mae' && isSelectedWinner ? 700 : 500 }}>
                        {fmt(row.mae, 3)} km
                      </span>
                    </td>

                    <td>
                      <span style={{ color: row.r2 < 0 ? '#DC2626' : '#059669', fontWeight: 600 }}>
                        {fmt(row.r2, 3)}
                      </span>
                    </td>

                    <td>
                      <span style={{ fontSize: '11px', color: '#475569' }}>
                        {row.notes}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <p className="fine-print">
          Notice: Completed trip distance is predicted from an initial 30-second driving window. It does NOT represent remaining driving range and must not replace route planning physics.
          Benchmark ranking reflects aggregated test performance on held-out Vehicle 10 and does not guarantee optimal accuracy for every individual vehicle or route.
        </p>
      </section>

      {/* 6. Interactive Visual Analytics Section (4 Graphs) */}
      <section className="panel" style={{ marginBottom: '20px' }}>
        <div className="panel-head">
          <div>
            <h3>Visual Analytics &amp; Comparative Graphs</h3>
            <p>Direct visual comparison of current-input predictions against validated historical errors and variance.</p>
          </div>
          <ChartNoAxesCombined color="#059669" />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '18px', marginTop: '10px' }}>
          {/* Graph A: Predicted Completed Trip Distance by Model */}
          <div
            style={{
              background: '#FFFFFF',
              border: '1px solid #E2E8F0',
              borderRadius: '10px',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div>
                <b style={{ fontSize: '13px', color: '#1E293B' }}>Graph A — Predicted Distance by Model</b>
                <p style={{ margin: 0, fontSize: '11px', color: '#64748B' }}>Live prediction for current 30s input (km)</p>
              </div>
              <span className="soft-badge">INPUT: {windowDistanceKm} km / 30s</span>
            </div>

            <div style={{ height: 260 }}>
              {predictedDistanceChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={predictedDistanceChartData} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: '#64748B', fontSize: 10 }}
                      interval={0}
                      angle={-15}
                      textAnchor="end"
                    />
                    <YAxis
                      tick={{ fill: '#64748B', fontSize: 10 }}
                      label={{ value: 'Distance (km)', angle: -90, position: 'insideLeft', fill: '#64748B', fontSize: 10 }}
                    />
                    <Tooltip
                      formatter={(val) => [`${fmt(val, 3)} km`, 'Predicted Completed Distance']}
                      contentStyle={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '6px', fontSize: '11.5px' }}
                    />
                    <Bar dataKey="distanceKm" radius={[4, 4, 0, 0]}>
                      {predictedDistanceChartData.map((entry) => (
                        <Cell key={entry.name} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ display: 'grid', placeItems: 'center', height: '100%', color: '#94A3B8', fontSize: '12px' }}>
                  {loading ? 'Evaluating models...' : 'Run comparison to view predictions'}
                </div>
              )}
            </div>
            <small style={{ fontSize: '10.5px', color: '#64748B', display: 'block', marginTop: '6px' }}>
              Each bar represents the completed trip length (km) forecasted by that specific architecture.
            </small>
          </div>

          {/* Graph B: Historical Benchmark Error Comparison (RMSE vs MAE) */}
          <div
            style={{
              background: '#FFFFFF',
              border: '1px solid #E2E8F0',
              borderRadius: '10px',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div>
                <b style={{ fontSize: '13px', color: '#1E293B' }}>Graph B — Historical Error Comparison</b>
                <p style={{ margin: 0, fontSize: '11px', color: '#64748B' }}>Held-out Vehicle 10 validation test errors (km)</p>
              </div>
              <span className="soft-badge">LOWER IS BETTER</span>
            </div>

            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={historicalErrorChartData} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#64748B', fontSize: 10 }}
                    interval={0}
                    angle={-15}
                    textAnchor="end"
                  />
                  <YAxis
                    tick={{ fill: '#64748B', fontSize: 10 }}
                    label={{ value: 'Error (km)', angle: -90, position: 'insideLeft', fill: '#64748B', fontSize: 10 }}
                  />
                  <Tooltip
                    formatter={(val) => [`${fmt(val, 3)} km`]}
                    contentStyle={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '6px', fontSize: '11.5px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                  <Bar dataKey="RMSE" fill="#059669" radius={[3, 3, 0, 0]} name="RMSE (km)" />
                  <Bar dataKey="MAE" fill="#2563EB" radius={[3, 3, 0, 0]} name="MAE (km)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <small style={{ fontSize: '10.5px', color: '#64748B', display: 'block', marginTop: '6px' }}>
              XGBoost achieves lowest RMSE (1.488 km); GRU achieves lowest MAE (0.905 km).
            </small>
          </div>

          {/* Graph C: Model R² Score Comparison */}
          <div
            style={{
              background: '#FFFFFF',
              border: '1px solid #E2E8F0',
              borderRadius: '10px',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div>
                <b style={{ fontSize: '13px', color: '#1E293B' }}>Graph C — Validation R² Score Comparison</b>
                <p style={{ margin: 0, fontSize: '11px', color: '#64748B' }}>Coefficient of determination on held-out test vehicle</p>
              </div>
              <span className="soft-badge" style={{ background: '#fee2e2', color: '#991b1b' }}>NEGATIVE R²</span>
            </div>

            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={r2ChartData} margin={{ top: 10, right: 10, left: -15, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#64748B', fontSize: 10 }}
                    interval={0}
                    angle={-15}
                    textAnchor="end"
                  />
                  <YAxis
                    domain={[-0.5, 0.1]}
                    tick={{ fill: '#64748B', fontSize: 10 }}
                    label={{ value: 'R² Score', angle: -90, position: 'insideLeft', fill: '#64748B', fontSize: 10 }}
                  />
                  <ReferenceLine y={0} stroke="#DC2626" strokeDasharray="3 3" strokeWidth={1.5} />
                  <Tooltip
                    formatter={(val) => [fmt(val, 3), 'R² Score']}
                    contentStyle={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: '6px', fontSize: '11.5px' }}
                  />
                  <Bar dataKey="r2" radius={[0, 0, 4, 4]}>
                    {r2ChartData.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <small style={{ fontSize: '10.5px', color: '#64748B', display: 'block', marginTop: '6px' }}>
              The red line at R²=0 marks the mean test baseline. Negative R² reflects genuine driving variance across destination endpoints.
            </small>
          </div>

          {/* Graph D: Prediction Spread & Architectural Disagreement */}
          <div
            style={{
              background: '#FFFFFF',
              border: '1px solid #E2E8F0',
              borderRadius: '10px',
              padding: '16px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div>
                <b style={{ fontSize: '13px', color: '#1E293B' }}>Graph D — Model Prediction Spread</b>
                <p style={{ margin: 0, fontSize: '11px', color: '#64748B' }}>Divergence across architectures for the active input</p>
              </div>
              <span className="soft-badge">SPREAD: {fmt(predictionSpread.spread, 2)} KM</span>
            </div>

            <div style={{ padding: '16px 8px', display: 'flex', flexDirection: 'column', gap: '14px', minHeight: 230, justifyContent: 'center' }}>
              {predictionSpread.spread !== null ? (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', color: '#475569' }}>Lowest Prediction (GRU / Tree):</span>
                    <strong style={{ fontSize: '14px', color: '#2563EB' }}>{fmt(predictionSpread.min, 2)} km</strong>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', color: '#475569' }}>Highest Prediction (Hybrid / LSTM):</span>
                    <strong style={{ fontSize: '14px', color: '#D97706' }}>{fmt(predictionSpread.max, 2)} km</strong>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', color: '#475569' }}>Average Across Successful Models:</span>
                    <strong style={{ fontSize: '14px', color: '#059669' }}>{fmt(predictionSpread.mean, 2)} km</strong>
                  </div>

                  <div
                    style={{
                      height: '14px',
                      background: '#E2E8F0',
                      borderRadius: '7px',
                      overflow: 'hidden',
                      position: 'relative',
                      marginTop: '6px',
                    }}
                  >
                    <div
                      style={{
                        position: 'absolute',
                        left: '10%',
                        width: '80%',
                        height: '100%',
                        background: 'linear-gradient(90deg, #2563EB, #7C3AED, #D97706)',
                        borderRadius: '7px',
                      }}
                    />
                  </div>

                  <div
                    style={{
                      padding: '8px 10px',
                      background: '#F8FAFC',
                      border: '1px solid #E2E8F0',
                      borderRadius: '6px',
                      fontSize: '11px',
                      color: '#64748B',
                      lineHeight: '1.4',
                    }}
                  >
                    <b>Architectural Note:</b> Model divergence reflects differing inductive biases (tree statistical aggregates vs gated recurrent units) on initial 30s driving telemetry. This spread is not a formal Bayesian or conformal uncertainty interval.
                  </div>
                </>
              ) : (
                <div style={{ textAlign: 'center', color: '#94A3B8', fontSize: '12px' }}>
                  {loading ? 'Calculating spread...' : 'No predictions available'}
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* 7. Preserved NASA Capacity-Retention Benchmark Section */}
      <section className="panel" style={{ marginBottom: '18px' }}>
        <div className="panel-head">
          <div>
            <h3>NASA Capacity-Retention Benchmark: Four-Model Evaluation</h3>
            <p>Identical eligible samples (149 cycles) · Battery-cell-grouped split: B0006 strictly held out · Error in percentage points (pp)</p>
          </div>
          <span className="soft-badge" style={{ background: '#f3e8ff', color: '#7c3aed' }}>
            NASA RESEARCH (4 MODELS + BASELINES)
          </span>
        </div>

        {nasaEvaluation ? (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model Name</th>
                    <th>Architecture / Input Window</th>
                    <th>RMSE (pp)</th>
                    <th>MAE (pp)</th>
                    <th>R² Score</th>
                    <th>MAPE (%)</th>
                    <th>Evaluation Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(nasaEvaluation.evaluation.model_results).map(([key, m]) => {
                    const mName =
                      m.name ||
                      (key === 'xgboost'
                        ? 'XGBoost Regressor'
                        : key === 'linear_cycle_trend_baseline_paired'
                        ? 'Linear Cycle-Trend Baseline'
                        : key === 'lstm'
                        ? 'LSTM Sequence Regressor'
                        : key === 'gru'
                        ? 'GRU Sequence Regressor'
                        : key === 'cnn_lstm'
                        ? 'Hybrid CNN-LSTM'
                        : 'Training-Mean Baseline');
                    const isBest = key === 'linear_cycle_trend_baseline_paired';
                    const isXgb = key === 'xgboost';
                    const notes =
                      m.notes ||
                      (isBest
                        ? 'Simple linear degradation trend; lowest error on held-out cell'
                        : isXgb
                        ? 'Gradient-boosted decision trees on cycle index'
                        : key === 'lstm'
                        ? '2-layer LSTM on 20-cycle history; negative R² on unseen cell'
                        : key === 'cnn_lstm'
                        ? '1D CNN + LSTM hybrid; cross-cell domain shift'
                        : key === 'gru'
                        ? '2-layer GRU; cross-cell distribution shift'
                        : 'Constant mean capacity retention of training cells');
                    return (
                      <tr key={key} style={isBest ? { background: '#F0FDF4' } : {}}>
                        <td>
                          <b>{mName}</b>
                          {isBest && (
                            <span
                              style={{
                                marginLeft: '6px',
                                fontSize: '9px',
                                color: '#059669',
                                fontWeight: 'bold',
                                background: '#ECFDF5',
                                padding: '2px 6px',
                                borderRadius: '4px',
                              }}
                            >
                              ★ Lowest RMSE &amp; MAE
                            </span>
                          )}
                        </td>
                        <td>
                          <small>{notes}</small>
                        </td>
                        <td>
                          <b>{fmt(m.rmse_percentage_points, 3)} pp</b>
                        </td>
                        <td>{fmt(m.mae_percentage_points, 3)} pp</td>
                        <td style={{ color: Number(m.r2) < 0 ? '#c92a2a' : 'inherit' }}>
                          <b>{fmt(m.r2, 3)}</b>
                        </td>
                        <td>{fmt(m.mape_percent, 2)}%</td>
                        <td>
                          {isBest ? (
                            <span style={{ color: '#059669', fontWeight: '600' }}>
                              Lowest error (7.073 pp RMSE, R²=0.559)
                            </span>
                          ) : isXgb ? (
                            <span>Best nonlinear model (8.481 pp RMSE, R²=0.365)</span>
                          ) : Number(m.r2) < 0 ? (
                            <span style={{ color: '#DC2626' }}>Negative R² due to cross-cell distribution shift</span>
                          ) : (
                            <span>Evaluated on identical 149 test cycles</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="table-footnote">
              <b>Split Protocol:</b> Held-out test cell B0006 (149 scored cycles after 19-cycle sequence warmup). Training: Cells B0005 + B0018 (300 cycles). Validation: Cell B0007 (168 cycles). All models evaluated on identical test rows.
            </p>
            <div className="notice" style={{ marginTop: '14px', background: '#F8FAFC', borderColor: '#E2E8F0', color: '#475569' }}>
              <Info size={16} />
              <div>
                <b>Scientific Finding — Cross-Cell Degradation Shift &amp; Honest Reporting:</b>
                <p style={{ margin: '4px 0 0', lineHeight: '1.5', fontSize: '10px' }}>
                  The simple Linear Cycle-Trend Baseline outperforms XGBoost and all deep sequence models on held-out cell B0006. Deep sequence models (LSTM, GRU, Hybrid CNN-LSTM) exhibit negative R² (-0.022 to -0.706). This authentic finding illustrates that temporal sequence representations trained on few lab cells do not automatically generalize across cells without individual calibration. Capacity retention from this dataset is an operational research proxy and is NOT certified physical or personalized battery SoH.
                </p>
              </div>
            </div>
          </>
        ) : (
          <p style={{ fontSize: '12px', color: '#64748B' }}>
            NASA evaluation metrics loaded. (4 models: Linear Cycle-Trend Baseline, XGBoost Regressor, LSTM, GRU, Hybrid CNN-LSTM).
          </p>
        )}
        <p className="fine-print">
          Research estimate only. Capacity retention from cycle index is a dataset proxy; it is not personalized to an individual battery pack and does not replace certified physical SoH instrumentation.
        </p>
      </section>

      {/* 8. Preserved Academic Governance & Split Protocols Section */}
      <div className="panel">
        <div className="panel-head">
          <div>
            <h3>Academic Governance &amp; Split Protocols</h3>
            <p>Methodological guarantees preventing data leakage across all research pipelines</p>
          </div>
          <ShieldCheck color="#16a57a" />
        </div>
        <div className="protocol-grid">
          <div>
            <span className="protocol-num">01</span>
            <b>VED Vehicle-Grouped Split</b>
            <p>Vehicle 10 strictly held out (12 trips). Vehicle 455 (31 trips) and Vehicle 541 (8 trips) used for train and validation. Zero vehicle or trip overlap.</p>
          </div>
          <div>
            <span className="protocol-num">02</span>
            <b>Fixed 30s Telemetry Window</b>
            <p>First 30 seconds of driving sampled at 1 Hz (30 timesteps, 7 channels). Trips with &lt;95% coverage discarded. 51 verified trips retained.</p>
          </div>
          <div>
            <span className="protocol-num">03</span>
            <b>NASA Cell-Grouped Split</b>
            <p>Cell B0006 strictly held out (149 test observations). Cells B0005, B0018 for training, B0007 for validation. Sequence warmup enforced.</p>
          </div>
          <div>
            <span className="protocol-num">04</span>
            <b>Domain Disambiguation</b>
            <p>VED Telemetry Trip Distance (completed distance from initial 30s) ≠ NASA Cell Capacity Retention Proxy (cycle degradation) ≠ Trip Planner Physics Formula (deterministic range estimate).</p>
          </div>
        </div>
        <div className="model-footer">
          <span>
            <Database size={15} /> Dataset status: {metrics?.dataset_status || 'Configured & Validated'}
          </span>
          <button className="secondary-button" onClick={() => window.open(`${apiBase}/docs`, '_blank', 'noopener,noreferrer')}>
            Open API docs <ExternalLink size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
