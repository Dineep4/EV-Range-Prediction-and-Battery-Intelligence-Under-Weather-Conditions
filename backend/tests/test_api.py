import joblib
import numpy as np
import pandas as pd
import httpx

import app.main as api
from fastapi.testclient import TestClient

client = TestClient(api.app)

def test_health():
    r = client.get('/api/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'
    assert r.json()['dataset_status'] == 'processed_not_training_ready'
    assert r.json()['models']['epa_vehicle_range']['status'] == 'available'
    assert r.json()['models']['battery_soh']['status'] == 'unavailable'


def test_vite_development_origin_is_allowed_by_cors():
    response = client.get(
        '/api/health',
        headers={'Origin': 'http://127.0.0.1:5174'},
    )

    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == (
        'http://127.0.0.1:5174'
    )


def test_model_status_distinguishes_epa_from_unavailable_soh():
    response = client.get('/api/models/status')

    assert response.status_code == 200
    models = response.json()['models']
    assert models['epa_vehicle_range']['status'] == 'available'
    assert models['epa_vehicle_range']['model'] == 'XGBoost'
    assert models['battery_soh']['status'] == 'unavailable'
    assert 'reference' in models['battery_soh']['reason'].lower()


def test_model_status_marks_epa_unavailable_when_artifact_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, 'MODELS', tmp_path)

    response = client.get('/api/models/status')

    assert response.status_code == 200
    models = response.json()['models']
    assert models['epa_vehicle_range']['status'] == 'unavailable'
    assert models['battery_soh']['status'] == 'unavailable'


def test_epa_vehicle_search_returns_predictable_source_configs():
    response = client.get('/api/vehicles/epa', params={'search': 'Nissan'})

    assert response.status_code == 200
    result = response.json()
    assert result['target_units'] == 'miles'
    assert result['results']
    vehicle = result['results'][0]
    assert vehicle['make'] == 'Nissan'
    assert vehicle['features']['year']
    assert list(vehicle['features']) == api.EPA_FEATURE_ORDER


def test_epa_vehicle_search_rejects_blank_query():
    response = client.get('/api/vehicles/epa', params={'search': ''})

    assert response.status_code == 422


def test_epa_vehicle_search_includes_ford_explorer_catalogue_record_18291():
    response = client.get(
        '/api/vehicles/epa',
        params={'search': '2002 Ford Explorer USPS Electric'},
    )

    assert response.status_code == 200
    matches = response.json()['results']
    vehicle = next(item for item in matches if item['source_record_id'] == 18291)
    assert vehicle['label'] == '2002 Ford Explorer USPS Electric'
    assert list(vehicle['features']) == api.EPA_FEATURE_ORDER
    assert vehicle['features']['combined_e_kwh_100_miles'] == 87


class FakeProviderResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_current_weather_returns_timestamped_provider_observations(monkeypatch):
    payload = {
        'current': {
            'temperature_2m': 28.5,
            'relative_humidity_2m': 62,
            'wind_speed_10m': 12.4,
            'precipitation': 0.2,
            'time': '2026-10-11T02:00',
        },
        'timezone': 'Asia/Kolkata',
    }
    monkeypatch.setattr(
        api.httpx,
        'get',
        lambda *args, **kwargs: FakeProviderResponse(payload),
    )

    response = client.get(
        '/api/weather/current',
        params={'latitude': 17.4, 'longitude': 78.3},
    )

    assert response.status_code == 200
    assert response.json() == {
        'temperature_c': 28.5,
        'relative_humidity_percent': 62.0,
        'wind_speed_kmh': 12.4,
        'precipitation_mm': 0.2,
        'observed_at': '2026-10-11T02:00',
        'timezone': 'Asia/Kolkata',
        'source': 'Open-Meteo current conditions',
        'is_live_observation': True,
    }


def test_current_weather_rejects_provider_failure_and_malformed_payload(
    monkeypatch,
):
    def provider_failure(*args, **kwargs):
        raise httpx.TimeoutException('provider timed out')

    monkeypatch.setattr(api.httpx, 'get', provider_failure)
    failed = client.get(
        '/api/weather/current',
        params={'latitude': 17.4, 'longitude': 78.3},
    )
    assert failed.status_code == 503
    assert failed.json() == {
        'status': 'unavailable',
        'source': 'Open-Meteo current conditions',
        'is_live_observation': False,
        'error': {
            'code': 'provider_request_failed',
            'message': 'The weather provider could not be reached.',
            'provider_error': 'provider timed out',
        },
    }

    monkeypatch.setattr(
        api.httpx,
        'get',
        lambda *args, **kwargs: FakeProviderResponse({'current': {}}),
    )
    malformed = client.get(
        '/api/weather/current',
        params={'latitude': 17.4, 'longitude': 78.3},
    )
    assert malformed.status_code == 503
    assert malformed.json()['status'] == 'unavailable'
    assert malformed.json()['is_live_observation'] is False
    assert malformed.json()['error']['code'] == 'incomplete_provider_response'


def test_current_weather_reports_provider_http_status(monkeypatch):
    def provider_rejected(*args, **kwargs):
        request = httpx.Request('GET', api.OPEN_METEO_URL)
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError(
            'provider rate limited request',
            request=request,
            response=response,
        )

    monkeypatch.setattr(api.httpx, 'get', provider_rejected)

    response = client.get(
        '/api/weather/current',
        params={'latitude': 17.4, 'longitude': 78.3},
    )

    assert response.status_code == 503
    assert response.json()['status'] == 'unavailable'
    assert response.json()['error']['code'] == 'provider_http_error'
    assert response.json()['error']['provider_status_code'] == 429


def test_charging_station_lookup_returns_mapped_records_without_availability(
    monkeypatch,
):
    monkeypatch.setattr(api, '_STATION_CACHE', {})
    provider_calls = []
    payload = {
        'elements': [
            {
                'type': 'node',
                'id': 123,
                'lat': 17.4,
                'lon': 78.3,
                'tags': {'name': 'Mapped charger', 'socket:type2': '2'},
            }
        ]
    }
    monkeypatch.setattr(
        api.httpx,
        'post',
        lambda *args, **kwargs: (
            provider_calls.append((args, kwargs))
            or FakeProviderResponse(payload)
        ),
    )

    response = client.post(
        '/api/charging-stations/near-route',
        json={
            'points': [{'latitude': 17.4, 'longitude': 78.3}],
            'radius_m': 4500,
        },
    )

    assert response.status_code == 200
    assert provider_calls[0][1]['headers']['Accept'] == 'application/json'
    assert provider_calls[0][1]['headers']['User-Agent'].startswith('VoltWise-')
    result = response.json()
    assert result['availability_status'] == 'not_provided_by_source'
    assert result['results'][0]['name'] == 'Mapped charger'
    assert result['results'][0]['connectors'] == {'type2': '2'}
    assert result['results'][0]['charging_power'] is None
    assert result['results'][0]['map_url'].startswith(
        'https://www.openstreetmap.org/'
    )


def test_charging_station_lookup_uses_route_distance_and_caches_provider_data(
    monkeypatch,
):
    monkeypatch.setattr(api, '_STATION_CACHE', {})
    calls = []
    payload = {
        'elements': [
            {
                'type': 'node',
                'id': 333,
                'lat': 0.001,
                'lon': 0.05,
                'tags': {
                    'socket:ccs': '2',
                    'charging_station:output': '50 kW',
                    'name': 'Along route',
                },
            },
            {
                'type': 'way',
                'id': 334,
                'center': {'lat': 0.02, 'lon': 0.05},
                'tags': {'operator': 'Mapped Operator'},
            },
        ]
    }
    monkeypatch.setattr(
        api.httpx,
        'post',
        lambda *args, **kwargs: (
            calls.append((args, kwargs)) or FakeProviderResponse(payload)
        ),
    )
    request = {
        'points': [
            {'latitude': 0, 'longitude': 0},
            {'latitude': 0, 'longitude': 0.1},
        ],
        'route_points': [
            {'latitude': 0, 'longitude': 0},
            {'latitude': 0, 'longitude': 0.05},
            {'latitude': 0, 'longitude': 0.1},
        ],
        'radius_m': 4500,
    }

    first = client.post('/api/charging-stations/near-route', json=request)
    second = client.post('/api/charging-stations/near-route', json=request)

    assert first.status_code == second.status_code == 200
    assert len(calls) == 1
    results = first.json()['results']
    assert [result['id'] for result in results] == [333, 334]
    assert results[0]['distance_from_route_m'] == 111
    assert results[0]['connectors'] == {'ccs': '2'}
    assert results[0]['charging_power'] == '50 kW'
    assert results[1]['name'] is None
    assert 'charging_power' in results[1] and results[1]['charging_power'] is None


def test_charging_station_provider_failure_is_explicit(monkeypatch):
    def provider_failure(*args, **kwargs):
        raise httpx.TimeoutException('provider timed out')

    monkeypatch.setattr(api.httpx, 'post', provider_failure)

    response = client.post(
        '/api/charging-stations/near-route',
        json={'points': [{'latitude': 17.4, 'longitude': 78.3}]},
    )

    assert response.status_code == 502


def test_charging_station_lookup_validates_coordinates():
    response = client.post(
        '/api/charging-stations/near-route',
        json={'points': [{'latitude': 95, 'longitude': 78.3}]},
    )

    assert response.status_code == 422


def test_route_geocodes_addresses_and_returns_real_osrm_geometry(monkeypatch):
    responses = [
        FakeProviderResponse([{
            'lat': '17.4',
            'lon': '78.3',
            'display_name': 'Start address, Hyderabad',
        }]),
        FakeProviderResponse([{
            'lat': '17.5',
            'lon': '78.4',
            'display_name': 'Destination address, Hyderabad',
        }]),
        FakeProviderResponse({
            'code': 'Ok',
            'routes': [{
                'distance': 12345.0,
                'duration': 3600.0,
                'geometry': {
                    'coordinates': [[78.3, 17.4], [78.35, 17.45], [78.4, 17.5]]
                },
            }],
        }),
    ]
    requests = []

    def fake_get(url, **kwargs):
        requests.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(api.httpx, 'get', fake_get)
    response = client.post(
        '/api/route',
        json={
            'start_address': 'Start address',
            'destination_address': 'Destination address',
        },
    )

    assert response.status_code == 200
    assert len(requests) == 3
    assert response.json()['distance_m'] == 12345
    assert response.json()['duration_s'] == 3600
    assert response.json()['start']['label'] == 'Start address, Hyderabad'
    assert response.json()['destination']['label'] == 'Destination address, Hyderabad'
    assert response.json()['geometry'] == [
        {'latitude': 17.4, 'longitude': 78.3},
        {'latitude': 17.45, 'longitude': 78.35},
        {'latitude': 17.5, 'longitude': 78.4},
    ]


def test_route_reports_invalid_address_and_provider_failure(monkeypatch):
    monkeypatch.setattr(
        api.httpx,
        'get',
        lambda *args, **kwargs: FakeProviderResponse([]),
    )
    missing = client.post(
        '/api/route',
        json={'start_address': 'Unknown place', 'destination_address': 'Other'},
    )
    assert missing.status_code == 422
    assert 'Address not found' in missing.json()['detail']

    def lookup_failure(*args, **kwargs):
        raise httpx.TimeoutException('routing provider timed out')

    monkeypatch.setattr(api.httpx, 'get', lookup_failure)
    failed = client.post(
        '/api/route',
        json={'start_address': 'Start place', 'destination_address': 'End place'},
    )
    assert failed.status_code == 502
    assert 'Address lookup service failed' in failed.json()['detail']


def test_route_reports_osrm_failure_and_invalid_provider_geometry(monkeypatch):
    responses = [
        FakeProviderResponse([{'lat': '17.4', 'lon': '78.3'}]),
        FakeProviderResponse([{'lat': '17.5', 'lon': '78.4'}]),
    ]

    def osrm_failure(url, **kwargs):
        if 'nominatim' in url:
            return responses.pop(0)
        raise httpx.TimeoutException('OSRM timed out')

    monkeypatch.setattr(api.httpx, 'get', osrm_failure)
    failed = client.post(
        '/api/route',
        json={'start_address': 'Start place', 'destination_address': 'End place'},
    )
    assert failed.status_code == 502
    assert 'Road routing service failed' in failed.json()['detail']

    responses = [
        FakeProviderResponse([{'lat': '17.4', 'lon': '78.3'}]),
        FakeProviderResponse([{'lat': '17.5', 'lon': '78.4'}]),
        FakeProviderResponse({
            'code': 'Ok',
            'routes': [{
                'distance': 1200,
                'duration': 300,
                'geometry': {'coordinates': [[78.3, 17.4], [999, 17.5]]},
            }],
        }),
    ]

    def malformed_geometry(url, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(api.httpx, 'get', malformed_geometry)
    malformed = client.post(
        '/api/route',
        json={'start_address': 'Start place', 'destination_address': 'End place'},
    )
    assert malformed.status_code == 502
    assert 'invalid route measurements' in malformed.json()['detail']


def test_epa_vehicle_range_resolves_canonical_catalogue_record_id():
    response = client.post(
        '/api/predict/epa-range/vehicle',
        json={'source_record_id': 18291},
    )

    assert response.status_code == 200
    assert response.json()['is_ml_prediction'] is True
    assert response.json()['range_type'] == (
        'standardized EPA vehicle/configuration estimate'
    )


def test_epa_vehicle_range_rejects_unknown_record_id():
    response = client.post(
        '/api/predict/epa-range/vehicle',
        json={'source_record_id': 999999999},
    )

    assert response.status_code == 404


def test_range_estimate():
    r = client.post('/api/predict/range', json={'battery_capacity_kwh':60,'soc_percent':80,'soh_percent':90,'consumption_wh_km':150})
    assert r.status_code == 200
    assert abs(r.json()['estimated_range_km'] - 288) < .01
    assert r.json()['is_ml_prediction'] is False

def test_dataset_status_reports_real_manifest():
    r = client.get('/api/datasets/status')
    assert r.status_code == 200
    assert r.json()['sample_count'] == 2040
    assert r.json()['excluded_observation_count'] == 3
    assert r.json()['review_flagged_row_count'] == 210
    assert r.json()['soh_target_status'] == 'provisional_not_ready_for_benchmark_training'

def test_model_performance_reports_no_fabricated_results():
    r = client.get('/api/model-performance')
    assert r.status_code == 200
    assert r.json()['dataset_status'] == 'processed_not_training_ready'
    epa = [
        experiment
        for experiment in r.json()['experiments']
        if experiment.get('task') == 'EPA vehicle-level range estimate'
    ]
    assert len(epa) == 1
    assert epa[0]['status'] == 'TRAINED_AND_EVALUATED'
    assert epa[0]['target_units'] == 'km'
    assert epa[0]['rmse'] > 0
    assert epa[0]['mae'] > 0
    assert epa[0]['r2'] is not None


class FixedRangeModel:
    def predict(self, features):
        return np.array([300.0])


class IdentityPreprocessor:
    def transform(self, features):
        return features


def epa_range_request():
    return {
        'year': 2025,
        'city_e_kwh_100_miles': 25,
        'highway_e_kwh_100_miles': 30,
        'combined_e_kwh_100_miles': 27,
        'charge_240_hours': 8,
        'drive': 'All-Wheel Drive',
        'vehicle_class': 'Small Sport Utility Vehicle',
    }


def test_epa_range_prediction_uses_evaluated_artifact_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'MODELS', tmp_path)
    joblib.dump(
        {
            'model': FixedRangeModel(),
            'preprocessor': IdentityPreprocessor(),
            'feature_order': api.EPA_FEATURE_ORDER,
            'target': 'epa_range_miles',
            'target_units': 'miles',
            'target_definition': 'EPA test range',
            'training_status': 'TRAINED_AND_EVALUATED',
        },
        tmp_path / 'epa_range_xgboost.joblib',
    )
    response = client.post('/api/predict/epa-range', json=epa_range_request())

    assert response.status_code == 200
    assert response.json()['predicted_epa_range_miles'] == 300
    assert response.json()['predicted_epa_range_km'] == 482.8
    assert response.json()['is_ml_prediction'] is True
    assert response.json()['range_type'] == (
        'standardized EPA vehicle/configuration estimate'
    )


def test_epa_range_prediction_is_unavailable_without_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'MODELS', tmp_path)

    response = client.post('/api/predict/epa-range', json=epa_range_request())

    assert response.status_code == 503


def test_epa_saved_artifact_loads_with_training_feature_schema():
    artifact = joblib.load(api.MODELS / 'epa_range_xgboost.joblib')

    assert artifact['training_status'] == 'TRAINED_AND_EVALUATED'
    assert artifact['target'] == 'epa_range_miles'
    assert artifact['target_units'] == 'miles'
    assert artifact['preprocessor'].feature_names_in_.tolist() == (
        artifact['feature_order']
    )
    assert artifact['feature_order'] == [
        'year',
        'city_e_kwh_100_miles',
        'highway_e_kwh_100_miles',
        'combined_e_kwh_100_miles',
        'charge_240_hours',
        'drive',
        'vehicle_class',
    ]
    assert artifact['model'].n_features_in_ == len(
        artifact['preprocessor'].get_feature_names_out()
    )


def test_epa_real_artifact_inference_uses_miles_and_kilometres_consistently():
    artifact = joblib.load(api.MODELS / 'epa_range_xgboost.joblib')
    features = pd.DataFrame(
        [epa_range_request()],
        columns=artifact['feature_order'],
    )
    expected_miles = float(artifact['model'].predict(
        artifact['preprocessor'].transform(features)
    )[0])
    response = client.post(
        '/api/predict/epa-range',
        json=epa_range_request(),
    )

    assert response.status_code == 200
    result = response.json()
    assert result['predicted_epa_range_miles'] == round(expected_miles, 2)
    assert result['predicted_epa_range_km'] == round(
        expected_miles * 1.609344, 2
    )
    assert result['is_ml_prediction'] is True


def test_epa_range_rejects_invalid_input():
    invalid = epa_range_request()
    invalid['year'] = 1800

    response = client.post('/api/predict/epa-range', json=invalid)

    assert response.status_code == 422


def test_epa_range_rejects_nonpositive_consumption_input():
    invalid = epa_range_request()
    invalid['combined_e_kwh_100_miles'] = 0

    response = client.post('/api/predict/epa-range', json=invalid)

    assert response.status_code == 422

def test_soh_unavailable_without_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'MODELS', tmp_path)
    r = client.post(
        '/api/predict/soh',
        json={
            'battery_id': 'cell-1',
            'cycle_index': 100,
            'capacity_ah': 2,
            'voltage_mean': 3.7,
            'current_mean': 1,
            'temperature_mean': 25,
        },
    )
    assert r.status_code == 503
    assert 'SOH prediction is unavailable' in r.json()['detail']


def test_soh_remains_unavailable_even_if_unvalidated_artifact_exists(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(api, 'MODELS', tmp_path)
    joblib.dump(
        {
            'model': FixedRangeModel(),
            'feature_order': ['cycle_index'],
            'target': 'soh_percent',
        },
        tmp_path / 'xgboost_soh.joblib',
    )
    r = client.post(
        '/api/predict/soh',
        json={
            'battery_id': 'cell-1',
            'cycle_index': 10,
            'capacity_ah': 2,
            'voltage_mean': 3.7,
            'current_mean': 1,
            'temperature_mean': 26,
        },
    )
    assert r.status_code == 503
    assert 'SOH prediction is unavailable' in r.json()['detail']

def test_range_rejects_invalid_inputs():
    # 0 or negative capacity
    r = client.post(
        '/api/predict/range',
        json={
            'battery_capacity_kwh': 0,
            'soc_percent': 80,
            'soh_percent': 90,
            'consumption_wh_km': 150,
        },
    )
    assert r.status_code == 422

    # SOC > 100
    r_soc = client.post(
        '/api/predict/range',
        json={
            'battery_capacity_kwh': 60,
            'soc_percent': 105,
            'soh_percent': 90,
            'consumption_wh_km': 150,
        },
    )
    assert r_soc.status_code == 422

    # Consumption <= 0
    r_cons = client.post(
        '/api/predict/range',
        json={
            'battery_capacity_kwh': 60,
            'soc_percent': 80,
            'soh_percent': 90,
            'consumption_wh_km': 0,
        },
    )
    assert r_cons.status_code == 422

    # SOH > 100
    r_soh = client.post(
        '/api/predict/range',
        json={
            'battery_capacity_kwh': 60,
            'soc_percent': 80,
            'soh_percent': 105,
            'consumption_wh_km': 150,
        },
    )
    assert r_soh.status_code == 422

def test_route_rejects_identical_or_short_locations():
    # Identical start and destination
    r_same = client.post(
        '/api/route',
        json={
            'start_address': 'Hyderabad',
            'destination_address': 'Hyderabad',
        },
    )
    assert r_same.status_code == 422
    assert 'Choose different starting and destination locations' in r_same.json()['detail']

    # Short start address
    r_short = client.post(
        '/api/route',
        json={
            'start_address': 'A',
            'destination_address': 'Secunderabad',
        },
    )
    assert r_short.status_code == 422

