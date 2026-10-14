import httpx
from fastapi.testclient import TestClient
import app.main as api

client = TestClient(api.app)

class ProviderResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", api.OVERPASS_URL)
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("provider error", request=request, response=response)
    def json(self):
        return self.payload

def request():
    return {"points": [{"latitude": 17.4, "longitude": 78.3}], "radius_m": 4500}

def test_station_response_identifies_live_then_cached_data_and_timestamp(monkeypatch):
    monkeypatch.setattr(api, "_STATION_CACHE", {})
    calls=[]
    payload={"elements":[{"type":"node","id":55,"lat":17.4,"lon":78.3,"tags":{"name":"Mapped site","access":"customers","opening_hours":"Mo-Su 00:00-24:00","fee":"yes","socket:type2":"2"}}]}
    monkeypatch.setattr(api.httpx,"post",lambda *a,**k:(calls.append(k) or ProviderResponse(payload)))
    first=client.post("/api/charging-stations/near-route",json=request())
    second=client.post("/api/charging-stations/near-route",json=request())
    assert first.status_code==second.status_code==200
    live=first.json(); cached=second.json()
    assert live["data_status"]=="live" and live["cached"] is False and live["retrieved_at"]
    assert cached["data_status"]=="cached" and cached["cached"] is True
    assert cached["retrieved_at"]==live["retrieved_at"] and len(calls)==1
    station=live["results"][0]
    assert station["access"]=="customers" and station["opening_hours"]=="Mo-Su 00:00-24:00" and station["fee"]=="yes"

def test_station_provider_retries_504_once_with_bounded_backoff(monkeypatch):
    monkeypatch.setattr(api,"_STATION_CACHE",{})
    monkeypatch.setattr(api,"OVERPASS_MAX_ATTEMPTS",2)
    sleeps=[]; calls=[]
    monkeypatch.setattr(api.time,"sleep",lambda delay:sleeps.append(delay))
    def provider(*args,**kwargs):
        calls.append(kwargs)
        return ProviderResponse({"elements": []},504) if len(calls)==1 else ProviderResponse({"elements": []})
    monkeypatch.setattr(api.httpx,"post",provider)
    response=client.post("/api/charging-stations/near-route",json=request())
    assert response.status_code==200 and response.json()["results"]==[]
    assert len(calls)==2 and len(sleeps)==1
    assert calls[0]["timeout"]==api.OVERPASS_TIMEOUT_SECONDS
    assert calls[0]["headers"]["User-Agent"].startswith("VoltWise-")

def test_station_provider_timeout_returns_safe_retryable_error_after_attempt_limit(monkeypatch):
    monkeypatch.setattr(api,"_STATION_CACHE",{})
    monkeypatch.setattr(api,"OVERPASS_MAX_ATTEMPTS",2)
    calls=[]; monkeypatch.setattr(api.time,"sleep",lambda _:None)
    def timeout(*args,**kwargs):
        calls.append(1); raise httpx.TimeoutException("private provider details")
    monkeypatch.setattr(api.httpx,"post",timeout)
    response=client.post("/api/charging-stations/near-route",json=request())
    assert response.status_code==502 and len(calls)==2
    assert response.json()["detail"]["code"]=="provider_unavailable"
    assert "private provider details" not in response.text

def test_station_provider_rejects_malformed_and_accepts_empty_results(monkeypatch):
    monkeypatch.setattr(api,"_STATION_CACHE",{})
    monkeypatch.setattr(api.httpx,"post",lambda *a,**k:ProviderResponse({"wrong": []}))
    malformed=client.post("/api/charging-stations/near-route",json=request())
    assert malformed.status_code==502
    monkeypatch.setattr(api,"_STATION_CACHE",{})
    monkeypatch.setattr(api.httpx,"post",lambda *a,**k:ProviderResponse({"elements": []}))
    empty=client.post("/api/charging-stations/near-route",json=request())
    assert empty.status_code==200 and empty.json()["results"]==[]
    assert empty.json()["availability_status"]=="not_provided_by_source"

def test_station_provider_serves_cached_fallback_on_504_when_cache_exists(monkeypatch):
    cache_key = (
        4500,
        (17.4, 78.3),
    )
    cached_stations = [{
        "id": 99,
        "osm_type": "node",
        "latitude": 17.4,
        "longitude": 78.3,
        "name": "Cached station",
        "operator": "Test Operator",
        "connectors": {"type2": "2"},
        "charging_power": "22 kW",
        "access": "public",
        "opening_hours": "24/7",
        "fee": "no",
        "map_url": "https://www.openstreetmap.org/#map=17/17.4/78.3",
    }]
    monkeypatch.setattr(api, "_STATION_CACHE", {
        cache_key: (0.0, cached_stations, "2026-10-12T00:00:00Z")
    })
    monkeypatch.setattr(api, "OVERPASS_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    monkeypatch.setattr(api.httpx, "post", lambda *a, **k: ProviderResponse({"elements": []}, 504))

    response = client.post("/api/charging-stations/near-route", json=request())
    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is True
    assert data["data_status"] == "cached"
    assert "cached fallback" in data["source"]
    assert data["notice"] is not None
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "Cached station"

def test_station_provider_query_omits_rel_and_limits_corridor_points(monkeypatch):
    monkeypatch.setattr(api, "_STATION_CACHE", {})
    captured_data = []
    def capture_post(*args, **kwargs):
        captured_data.append(kwargs.get("data", {}).get("data", ""))
        return ProviderResponse({"elements": []})
    monkeypatch.setattr(api.httpx, "post", capture_post)

    multi_request = {
        "points": [{"latitude": 17.4 + i * 0.01, "longitude": 78.3 + i * 0.01} for i in range(8)],
        "radius_m": 4500,
    }
    response = client.post("/api/charging-stations/near-route", json=multi_request)
    assert response.status_code == 200
    assert len(captured_data) == 1
    query = captured_data[0]
    assert "node[amenity=charging_station]" in query
    assert "way[amenity=charging_station]" in query
    assert "rel[amenity=charging_station]" not in query
    assert "nwr[amenity=charging_station]" not in query

def test_station_provider_504_without_cache_returns_retryable_502(monkeypatch):
    monkeypatch.setattr(api, "_STATION_CACHE", {})
    monkeypatch.setattr(api, "OVERPASS_MAX_ATTEMPTS", 2)
    sleeps = []
    calls = []
    monkeypatch.setattr(api.time, "sleep", lambda delay: sleeps.append(delay))
    def always_504(*args, **kwargs):
        calls.append(kwargs)
        return ProviderResponse({"elements": []}, 504)
    monkeypatch.setattr(api.httpx, "post", always_504)
    response = client.post("/api/charging-stations/near-route", json=request())
    assert response.status_code == 502
    assert len(calls) == 2
    assert len(sleeps) == 1
    data = response.json()
    assert data["detail"]["code"] == "provider_unavailable"
    assert data["detail"]["retryable"] is True
    assert "HTTP 504" in data["detail"]["message"]


