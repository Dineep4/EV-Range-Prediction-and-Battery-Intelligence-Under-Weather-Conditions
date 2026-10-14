import React, { useState, useEffect, useMemo } from 'react';
import {
  MapPin,
  Search,
  RefreshCw,
  AlertTriangle,
  Info,
  ExternalLink,
  Zap,
  Sliders,
  CheckCircle2,
  ShieldAlert,
  HelpCircle,
  Compass,
} from 'lucide-react';

import {
  generateSimulatedChargingStations,
  resolveSourceLocation,
  isValidCoordinate,
  DEMO_MODE_BANNER,
  SIMULATED_DISCLAIMER,
} from './simulatedChargingStations.js';

export default function ChargingStationsView({
  route = null,
  availableRangeKm = null,
  MapPanelComponent,
}) {
  // Source location priority inputs
  const [sourceAddress, setSourceAddress] = useState('');
  const [sourceLatInput, setSourceLatInput] = useState('');
  const [sourceLonInput, setSourceLonInput] = useState('');
  const [activeSourceLabel, setActiveSourceLabel] = useState('');
  const [searchRadius, setSearchRadius] = useState(8000); // 8 km

  const [geocodingLoading, setGeocodingLoading] = useState(false);
  const [geocodingError, setGeocodingError] = useState('');

  // Selected station for details view
  const [selectedStation, setSelectedStation] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Initialize or resolve source location based on priority:
  // 1. Current user input coordinates
  // 2. Trip Planner origin (route.start)
  // 3. Default (Hyderabad)
  const resolvedSource = useMemo(() => {
    return resolveSourceLocation({
      searchLat: sourceLatInput,
      searchLon: sourceLonInput,
      searchLabel: activeSourceLabel || sourceAddress,
      routeStart: route?.start,
      routeStartLabel: route?.startLabel,
      defaultLocation: {
        latitude: 17.4474,
        longitude: 78.3488,
        label: 'Default Location: Hyderabad (17.4474, 78.3488)',
      },
    });
  }, [sourceLatInput, sourceLonInput, activeSourceLabel, sourceAddress, route, refreshTrigger]);

  // Synchronize initial input fields if empty
  useEffect(() => {
    if (!sourceLatInput && !sourceLonInput) {
      if (route?.start && Array.isArray(route.start)) {
        setSourceLatInput(String(route.start[0]));
        setSourceLonInput(String(route.start[1]));
        setActiveSourceLabel(route.startLabel || 'Trip Planner Origin');
        setSourceAddress(route.startLabel || '');
      } else {
        setSourceLatInput('17.4474');
        setSourceLonInput('78.3488');
        setActiveSourceLabel('Hyderabad (Default Demo Location)');
      }
    }
  }, [route]);

  // Generate 3-4 simulated charging stations deterministically
  const simulatedStations = useMemo(() => {
    if (!isValidCoordinate(resolvedSource.latitude, resolvedSource.longitude)) {
      return [];
    }
    return generateSimulatedChargingStations({
      sourceLat: resolvedSource.latitude,
      sourceLon: resolvedSource.longitude,
      searchRadiusM: searchRadius,
      count: 4,
      route,
      availableRangeKm,
    });
  }, [resolvedSource, searchRadius, route, availableRangeKm, refreshTrigger]);

  // Handle address geocoding search (e.g. "Bengaluru", "Connaught Place", "Banjara Hills")
  const handleAddressSearch = async (e) => {
    if (e && typeof e.preventDefault === 'function') e.preventDefault();
    const query = sourceAddress.trim();
    if (!query) return;

    setGeocodingLoading(true);
    setGeocodingError('');
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(query)}`,
        {
          headers: {
            'Accept': 'application/json',
            'Accept-Language': 'en',
          },
        }
      );
      if (!res.ok) throw new Error('Geocoding service unavailable.');
      const data = await res.json();
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error(`Location not found: "${query}". Try adding city or country.`);
      }
      const item = data[0];
      const lat = parseFloat(item.lat);
      const lon = parseFloat(item.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        throw new Error('Invalid coordinate received from location service.');
      }
      setSourceLatInput(lat.toFixed(4));
      setSourceLonInput(lon.toFixed(4));
      setActiveSourceLabel(item.display_name?.split(',')[0] || query);
      setSelectedStation(null);
    } catch (err) {
      setGeocodingError(err.message || 'Could not find address.');
    } finally {
      setGeocodingLoading(false);
    }
  };

  // Quick preset cities for easy testing & demo
  const presetCities = [
    { name: 'Hyderabad', lat: '17.4474', lon: '78.3488' },
    { name: 'Bengaluru', lat: '12.9716', lon: '77.5946' },
    { name: 'Mumbai', lat: '19.0760', lon: '72.8777' },
    { name: 'New Delhi', lat: '28.6139', lon: '77.2090' },
  ];

  const handleApplyPreset = (city) => {
    setSourceLatInput(city.lat);
    setSourceLonInput(city.lon);
    setSourceAddress(city.name);
    setActiveSourceLabel(`${city.name} Demo Center`);
    setGeocodingError('');
    setSelectedStation(null);
    setRefreshTrigger((prev) => prev + 1);
  };

  const handleRefreshStations = () => {
    setSelectedStation(null);
    setRefreshTrigger((prev) => prev + 1);
  };

  const fmt = (n, d = 1) =>
    n === null || n === undefined || n === '' || !Number.isFinite(Number(n)) ? '—' : Number(n).toFixed(d);

  return (
    <div className="charging-stations-container">
      {/* 1. Header Toolbar */}
      <div className="charging-toolbar">
        <div>
          <h3>Interactive Simulated EV Charging Stations</h3>
          <p>
            Displays 3–4 simulated demonstration charging points with realistic geographic placement and range-aware recommendations.
          </p>
        </div>
        <button
          className="primary-button"
          onClick={handleRefreshStations}
          title="Regenerate or refresh simulated stations"
        >
          <RefreshCw size={16} /> Find nearby stations
        </button>
      </div>

      {/* 2. Prominent Honest Demo Banner */}
      <div
        className="notice warning-notice"
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: '10px',
          marginBottom: '16px',
          background: '#fef3c7',
          border: '1px solid #f59e0b',
          borderRadius: '8px',
          padding: '12px 16px',
          color: '#92400e',
          lineHeight: '1.5',
        }}
      >
        <Info size={18} style={{ flexShrink: 0, marginTop: '2px', color: '#b45309' }} />
        <div>
          <strong>Academic Demonstration Mode:</strong> {DEMO_MODE_BANNER}
          <div style={{ marginTop: '4px', fontSize: '11.5px', color: '#b45309' }}>
            {SIMULATED_DISCLAIMER} No real-time availability, commercial pricing, or certified connector compatibility is represented.
          </div>
        </div>
      </div>

      {/* 3. Source Location Search & Radius Controls */}
      <div className="panel station-search-controls" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
          <h4 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#1e293b' }}>
            <MapPin size={16} style={{ display: 'inline', verticalAlign: '-2px', marginRight: '6px', color: '#2563eb' }} />
            Source Location &amp; Search Radius
          </h4>
          <span className="soft-badge" style={{ fontSize: '11px', textTransform: 'uppercase' }}>
            {resolvedSource.sourceType.replace('_', ' ')}
          </span>
        </div>

        {/* Address lookup form */}
        <form onSubmit={handleAddressSearch} style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
          <div className="input-with-icon" style={{ flex: 1 }}>
            <Search size={16} />
            <input
              type="text"
              value={sourceAddress}
              onChange={(e) => setSourceAddress(e.target.value)}
              placeholder="Search any city or address (e.g. Bengaluru, Mumbai, Connaught Place)..."
            />
          </div>
          <button type="submit" className="secondary-button" disabled={geocodingLoading}>
            {geocodingLoading ? <RefreshCw className="spin" size={15} /> : 'Search Address'}
          </button>
        </form>

        {geocodingError && (
          <div className="error-box" style={{ marginBottom: '10px' }}>
            <AlertTriangle size={15} /> {geocodingError}
          </div>
        )}

        {/* Quick presets */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
          <span style={{ fontSize: '11.5px', color: '#64748b', fontWeight: 600 }}>Quick Presets:</span>
          {presetCities.map((city) => (
            <button
              key={city.name}
              type="button"
              className="text-button"
              style={{
                fontSize: '11.5px',
                padding: '3px 8px',
                borderRadius: '4px',
                background: sourceAddress.toLowerCase().includes(city.name.toLowerCase()) ? '#e0e7ff' : '#f1f5f9',
                color: sourceAddress.toLowerCase().includes(city.name.toLowerCase()) ? '#3730a3' : '#475569',
              }}
              onClick={() => handleApplyPreset(city)}
            >
              {city.name}
            </button>
          ))}
        </div>

        {/* Lat, Lon, Radius form */}
        <div className="form-row">
          <label>
            Latitude
            <input
              type="number"
              step="0.0001"
              value={sourceLatInput}
              onChange={(e) => {
                setSourceLatInput(e.target.value);
                setActiveSourceLabel('');
              }}
              placeholder="e.g. 17.4474"
            />
          </label>
          <label>
            Longitude
            <input
              type="number"
              step="0.0001"
              value={sourceLonInput}
              onChange={(e) => {
                setSourceLonInput(e.target.value);
                setActiveSourceLabel('');
              }}
              placeholder="e.g. 78.3488"
            />
          </label>
          <label>
            Search Radius
            <select value={searchRadius} onChange={(e) => setSearchRadius(Number(e.target.value))}>
              <option value={3000}>3 km</option>
              <option value={5000}>5 km</option>
              <option value={8000}>8 km (Default)</option>
              <option value={12000}>12 km</option>
              <option value={15000}>15 km</option>
            </select>
          </label>
        </div>

        <div style={{ marginTop: '8px', fontSize: '12px', color: '#64748b' }}>
          Current Active Center: <b>{resolvedSource.label}</b> ({fmt(resolvedSource.latitude, 4)}, {fmt(resolvedSource.longitude, 4)})
        </div>
      </div>

      {/* 4. Interactive Map Panel */}
      <div className="panel charging-map" style={{ marginBottom: '16px' }}>
        <MapPanelComponent
          center={[resolvedSource.latitude, resolvedSource.longitude]}
          route={route}
          routePoints={route?.points || []}
          stations={simulatedStations}
          sourceMarker={{
            latitude: resolvedSource.latitude,
            longitude: resolvedSource.longitude,
            label: resolvedSource.label,
          }}
          height={460}
          onStationSelect={setSelectedStation}
        />

        {/* Enhanced Map Legend */}
        <div className="map-legend">
          <span>
            <i className="legend-dot" style={{ background: '#2563eb' }} /> Source Location
          </span>
          {route && (
            <span>
              <i className="legend-dot blue-dot" /> Planned Route
            </span>
          )}
          <span>
            <i className="legend-dot" style={{ background: '#7c3aed' }} /> Simulated EV Charging Point (⚡)
          </span>
          <span style={{ marginLeft: 'auto', fontWeight: 600, color: '#475569' }}>
            {simulatedStations.length} simulated point{simulatedStations.length === 1 ? '' : 's'} displayed
          </span>
        </div>
      </div>

      {/* 5. Selected Station Detailed Inspector Modal / Panel */}
      {selectedStation && (
        <section
          className="panel station-detail"
          style={{
            marginBottom: '16px',
            border: '2px solid #7c3aed',
            background: '#faf5ff',
          }}
        >
          <div className="panel-head">
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span
                  style={{
                    background: '#7c3aed',
                    color: 'white',
                    fontSize: '11px',
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: '4px',
                  }}
                >
                  SIMULATED
                </span>
                <h3 style={{ margin: 0, color: '#4c1d95' }}>{selectedStation.name}</h3>
              </div>
              <p style={{ margin: 0, color: '#6b21a8' }}>
                {selectedStation.statusLabel} · Coordinates: {fmt(selectedStation.lat, 5)}, {fmt(selectedStation.lon, 5)}
              </p>
            </div>
            <button
              className="icon-button"
              onClick={() => setSelectedStation(null)}
              aria-label="Close station details"
            >
              Close
            </button>
          </div>

          <div
            className="station-info"
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '12px',
              padding: '12px 0',
            }}
          >
            <div>
              <span style={{ color: '#64748b', fontSize: '11px', display: 'block' }}>ESTIMATED DISTANCE</span>
              <strong style={{ fontSize: '14px', color: '#1e293b' }}>
                Approx. {fmt(selectedStation.distanceFromSourceKm, 1)} km from source
              </strong>
              {Number.isFinite(selectedStation.routeProgressKm) && (
                <div style={{ fontSize: '11.5px', color: '#7c3aed' }}>
                  Progress along route: {fmt(selectedStation.routeProgressKm, 1)} km
                </div>
              )}
            </div>

            <div>
              <span style={{ color: '#64748b', fontSize: '11px', display: 'block' }}>CONNECTORS &amp; CHARGING</span>
              <strong style={{ fontSize: '14px', color: '#1e293b' }}>
                {selectedStation.chargingPower}
              </strong>
              <div style={{ fontSize: '11.5px', color: '#475569' }}>
                Types: {selectedStation.connectorTypes}
              </div>
            </div>

            <div>
              <span style={{ color: '#64748b', fontSize: '11px', display: 'block' }}>SIMULATED ACCESS &amp; HOURS</span>
              <strong style={{ fontSize: '13px', color: '#1e293b' }}>
                {selectedStation.openingHours}
              </strong>
              <div style={{ fontSize: '11.5px', color: '#475569' }}>
                {selectedStation.access} · {selectedStation.fee}
              </div>
            </div>

            <div>
              <span style={{ color: '#64748b', fontSize: '11px', display: 'block' }}>RECOMMENDATION STATUS</span>
              <span
                style={{
                  display: 'inline-block',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  fontSize: '11.5px',
                  fontWeight: 600,
                  background:
                    selectedStation.recommendationStatus === 'recommended_stop'
                      ? '#dcfce7'
                      : selectedStation.recommendationStatus === 'unreachable'
                      ? '#fee2e2'
                      : '#f1f5f9',
                  color:
                    selectedStation.recommendationStatus === 'recommended_stop'
                      ? '#166534'
                      : selectedStation.recommendationStatus === 'unreachable'
                      ? '#991b1b'
                      : '#334155',
                }}
              >
                {selectedStation.recommendationBadge}
              </span>
              <div style={{ fontSize: '11.5px', color: '#475569', marginTop: '2px' }}>
                {selectedStation.recommendationReason}
              </div>
            </div>
          </div>

          <div
            style={{
              padding: '8px 12px',
              background: '#f3e8ff',
              borderRadius: '6px',
              fontSize: '11.5px',
              color: '#581c87',
              marginTop: '8px',
            }}
          >
            {SIMULATED_DISCLAIMER}
          </div>
        </section>
      )}

      {/* 6. List of Simulated Station Cards */}
      <div className="station-list">
        {simulatedStations.map((station, index) => {
          const isSelected = selectedStation?.id === station.id;
          return (
            <div
              className={`station-row ${isSelected ? 'selected-row' : ''}`}
              key={station.id}
              style={{
                borderLeft: isSelected ? '4px solid #7c3aed' : '4px solid transparent',
                background: isSelected ? '#faf5ff' : undefined,
                transition: 'all 0.2s ease',
              }}
            >
              <div
                className="station-icon"
                style={{
                  background: isSelected ? '#7c3aed' : '#f3e8ff',
                  color: isSelected ? '#ffffff' : '#7c3aed',
                  width: '38px',
                  height: '38px',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Zap size={20} />
              </div>

              <div className="station-info" style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  <b style={{ fontSize: '14px', color: '#1e293b' }}>{station.name}</b>
                  <span
                    style={{
                      background: '#ede9fe',
                      color: '#6d28d9',
                      fontSize: '11px',
                      fontWeight: 600,
                      padding: '1px 6px',
                      borderRadius: '4px',
                    }}
                  >
                    Simulated location
                  </span>
                  <span
                    style={{
                      background:
                        station.recommendationStatus === 'recommended_stop'
                          ? '#dcfce7'
                          : station.recommendationStatus === 'unreachable'
                          ? '#fee2e2'
                          : '#f1f5f9',
                      color:
                        station.recommendationStatus === 'recommended_stop'
                          ? '#166534'
                          : station.recommendationStatus === 'unreachable'
                          ? '#991b1b'
                          : '#475569',
                      fontSize: '11px',
                      fontWeight: 600,
                      padding: '1px 6px',
                      borderRadius: '4px',
                    }}
                  >
                    {station.recommendationBadge}
                  </span>
                </div>

                <div style={{ margin: '3px 0', fontSize: '12.5px', color: '#334155' }}>
                  <span>{station.connectorTypes}</span> · <span>{station.chargingPower}</span> ·{' '}
                  <span style={{ fontWeight: 600 }}>Approx. {fmt(station.distanceFromSourceKm, 1)} km from source</span>
                </div>

                <div style={{ fontSize: '11.5px', color: '#64748b' }}>
                  {station.recommendationReason}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => setSelectedStation(station)}
                >
                  {isSelected ? 'Viewing' : 'Details'}
                </button>
                <button
                  type="button"
                  className="icon-button"
                  title="View on OpenStreetMap"
                  onClick={() => window.open(station.mapUrl, '_blank', 'noopener,noreferrer')}
                >
                  <ExternalLink size={15} />
                </button>
              </div>
            </div>
          );
        })}

        {simulatedStations.length === 0 && (
          <div className="empty-state">
            <Zap size={28} color="#7c3aed" />
            <b>No valid source coordinates</b>
            <p>Enter valid coordinates or an address to generate 3–4 simulated charging locations.</p>
          </div>
        )}
      </div>
    </div>
  );
}
