from __future__ import annotations
import json, os, sys
import pickle
from pathlib import Path
from typing import Any
import math
import time
from datetime import datetime, timezone
import joblib
import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "dataset_manifest.json"
EPA_DATA = ROOT / "data" / "processed" / "epa" / "epa_ev_range.csv"
OPEN_METEO_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
OSRM_URL = os.getenv("OSRM_URL", "https://router.project-osrm.org/route/v1/driving")

def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(max(value, minimum), maximum)

def _bounded_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)

OVERPASS_TIMEOUT_SECONDS = _bounded_env_float("OVERPASS_TIMEOUT_SECONDS", 15.0, 3.0, 30.0)
OVERPASS_MAX_ATTEMPTS = _bounded_env_int("OVERPASS_MAX_ATTEMPTS", 2, 1, 3)
OVERPASS_RETRY_BACKOFF_SECONDS = _bounded_env_float("OVERPASS_RETRY_BACKOFF_SECONDS", 0.4, 0.0, 2.0)
STATION_CACHE_TTL_SECONDS = _bounded_env_int("STATION_CACHE_TTL_SECONDS", 300, 30, 1800)
EPA_FEATURE_ORDER = [
    "year",
    "city_e_kwh_100_miles",
    "highway_e_kwh_100_miles",
    "combined_e_kwh_100_miles",
    "charge_240_hours",
    "drive",
    "vehicle_class",
]
SOH_UNAVAILABLE_REASON = (
    "SOH prediction is unavailable without a validated model artifact. "
    "A provisional reference model is available via research endpoints."
)

NASA_RESEARCH_REGISTRY = ROOT / "backend" / "nasa_research_model_registry.json"
NASA_FEATURE_ORDER = ["cycle_index_in_file"]
app = FastAPI(title="VoltWise EV Intelligence API", version="1.0.0", description="EV range planning, battery SOH prediction and experiment-result endpoints.")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174").split(",")], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class RangeRequest(BaseModel):
    battery_capacity_kwh: float = Field(gt=0, le=300)
    soc_percent: float = Field(ge=0, le=100)
    soh_percent: float = Field(gt=0, le=100)
    consumption_wh_km: float = Field(gt=0, le=2000)
    weather_factor: float = Field(default=1.0, ge=0.5, le=1.2)

class SOHRequest(BaseModel):
    battery_id: str = "unknown"
    cycle_index: float = Field(ge=0)
    capacity_ah: float | None = Field(default=None, gt=0)
    voltage_mean: float = Field(gt=0, le=10)
    current_mean: float = Field(ge=-1000, le=1000)
    temperature_mean: float = Field(ge=-100, le=200)
    rated_capacity_ah: float | None = Field(default=2.0, gt=0, le=5.0)

class NasaRetentionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cycle_index_in_file: int = Field(strict=True, ge=2, le=614)

class EPARangeRequest(BaseModel):
    year: int = Field(ge=1984, le=2100)
    city_e_kwh_100_miles: float = Field(gt=0, le=500)
    highway_e_kwh_100_miles: float = Field(gt=0, le=500)
    combined_e_kwh_100_miles: float = Field(gt=0, le=500)
    charge_240_hours: float = Field(ge=0, le=1000)
    drive: str = Field(min_length=1, max_length=100)
    vehicle_class: str = Field(min_length=1, max_length=100)

class EPAVehicleRangeRequest(BaseModel):
    source_record_id: int = Field(gt=0)

class RouteRequest(BaseModel):
    start_address: str = Field(min_length=2, max_length=300)
    destination_address: str = Field(min_length=2, max_length=300)

class RouteSampleRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

class ChargingStationsRequest(BaseModel):
    points: list[RouteSampleRequest] = Field(min_length=1, max_length=10)
    route_points: list[RouteSampleRequest] = Field(default_factory=list, max_length=1000)
    radius_m: int = Field(default=4500, ge=100, le=20000)

TRIP_DISTANCE_TABULAR_FEATURES = [
    "window_duration_s",
    "window_distance_km",
    "speed_mean",
    "speed_max",
    "speed_min",
    "speed_std",
    "power_mean",
    "power_max",
    "power_min",
    "current_mean",
    "voltage_mean",
    "soc_start",
    "soc_current",
    "soc_delta",
    "temperature_mean",
]

class TripDistancePredictRequest(BaseModel):
    model_name: str = Field(default="XGBoost")
    window_distance_km: float = Field(default=0.25, ge=0, le=50)
    speed_mean: float = Field(default=30.0, ge=0, le=200)
    speed_max: float = Field(default=45.0, ge=0, le=200)
    speed_min: float = Field(default=0.0, ge=0, le=200)
    speed_std: float = Field(default=10.0, ge=0, le=100)
    power_mean: float = Field(default=8.5, ge=-200, le=300)
    power_max: float = Field(default=25.0, ge=-200, le=300)
    power_min: float = Field(default=-10.0, ge=-200, le=300)
    current_mean: float = Field(default=22.0, ge=-500, le=500)
    voltage_mean: float = Field(default=385.0, ge=200, le=500)
    soc_start: float = Field(default=80.0, ge=0, le=100)
    soc_current: float = Field(default=79.8, ge=0, le=100)
    soc_delta: float = Field(default=-0.2, ge=-100, le=100)
    temperature_mean: float = Field(default=20.0, ge=-40, le=60)
    sequence: list[list[float]] | None = Field(default=None)

_STATION_CACHE: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]], str]] = {}

@app.get("/")
def root():
    return {"name": "VoltWise EV Intelligence API", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
def health():
    models = _models_status()
    return {
        "status": "ok",
        "service": "VoltWise API",
        "soh_models_available": [],
        "research_capacity_retention_available": (
            models["nasa_capacity_retention_proxy"]["status"] == "available"
        ),
        "range_models_available": (
            ["epa_range_xgboost.joblib"]
            if models["epa_vehicle_range"]["status"] == "available"
            else []
        ),
        "epa_range_model_available": (
            models["epa_vehicle_range"]["status"] == "available"
        ),
        "trip_distance_research_available": (
            models.get("trip_distance_research", {}).get("status") == "available"
        ),
        "models": models,
        "metrics_available": (
            (RESULTS / "model_metrics.json").exists()
            or (RESULTS / "epa_range_metrics.json").exists()
            or (RESULTS / "trip_distance_metrics.json").exists()
        ),
        "dataset_status": dataset_status()["status"],
    }


def _load_epa_artifact() -> dict[str, Any]:
    model_path = MODELS / "epa_range_xgboost.joblib"
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="No evaluated EPA vehicle-range model is installed.",
        )
    try:
        artifact = joblib.load(model_path)
    except (OSError, EOFError, ImportError, ValueError, pickle.UnpicklingError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load EPA range model artifact: {exc}",
        ) from exc

    if (
        not isinstance(artifact, dict)
        or artifact.get("training_status") != "TRAINED_AND_EVALUATED"
        or artifact.get("target") != "epa_range_miles"
        or artifact.get("target_units") != "miles"
        or artifact.get("feature_order") != EPA_FEATURE_ORDER
        or not isinstance(artifact.get("target_definition"), str)
    ):
        raise HTTPException(
            status_code=503,
            detail="EPA range model artifact is missing compatible evaluation metadata.",
        )
    if not callable(getattr(artifact.get("model"), "predict", None)) or not callable(
        getattr(artifact.get("preprocessor"), "transform", None)
    ):
        raise HTTPException(
            status_code=503,
            detail="EPA range model artifact does not contain compatible inference objects.",
        )
    return artifact


def _models_status() -> dict[str, dict[str, str]]:
    try:
        _load_epa_artifact()
    except HTTPException as exc:
        epa_status = {
            "status": "unavailable",
            "reason": exc.detail,
        }
    else:
        epa_status = {
            "status": "available",
            "model": "XGBoost",
            "artifact": "models/epa_range_xgboost.joblib",
            "target": "EPA standardized vehicle/configuration range",
            "target_units": "miles",
            "prediction_endpoint": "/api/predict/epa-range",
        }
    try:
        nasa_config, _ = _load_nasa_retention_model()
    except HTTPException as exc:
        nasa_status = {"status": "unavailable", "reason": exc.detail}
    else:
        nasa_status = {
            "status": "available",
            "dataset": nasa_config["dataset"],
            "model": nasa_config["model"]["name"],
            "target": nasa_config["target"]["name"],
            "artifact": nasa_config["model"]["artifact_path"],
            "prediction_endpoint": "/api/predict/nasa-capacity-retention",
            "provisional_proxy": True,
        }

    trip_distance_artifact = MODELS / "trip_distance_xgboost.joblib"
    if trip_distance_artifact.is_file():
        trip_status = {
            "status": "available",
            "models": ["XGBoost", "LSTM", "GRU", "Hybrid CNN-LSTM", "Training-Mean Baseline"],
            "target": "total_trip_distance_km",
            "target_units": "km",
            "observation_window": "First 30 seconds at 1 Hz",
            "prediction_endpoint": "/api/predict/trip-distance",
            "dataset": "Vehicle Energy Dataset (pure-EV 2013 Nissan Leafs)",
            "research_estimate": True,
            "is_research_estimate": True,
        }
    else:
        trip_status = {
            "status": "unavailable",
            "reason": "Trip distance model artifacts not found.",
        }

    soh_model_artifact = MODELS / "battery_soh_model.joblib"
    if soh_model_artifact.is_file():
        soh_model_status = {
            "status": "available",
            "model": "Gradient Boosting Regressor",
            "dataset": "NASA PCoE Li-ion Battery Aging Dataset (2.0 Ah 18650 cells)",
            "target": "SOH (% of 2.0 Ah rated capacity)",
            "target_definition": "100 * measured discharge capacity / 2.0 Ah rated capacity",
            "prediction_endpoint": "/api/predict/soh",
            "test_cell": "B0006 (unseen held-out cell)",
        }
    else:
        soh_model_status = {
            "status": "unavailable",
            "reason": "Battery SOH model artifact not found.",
        }

    return {
        "epa_vehicle_range": epa_status,
        "battery_soh": {
            "status": "unavailable",
            "reason": SOH_UNAVAILABLE_REASON,
        },
        "battery_soh_model": soh_model_status,
        "nasa_capacity_retention_proxy": nasa_status,
        "trip_distance_research": trip_status,
    }


@app.get("/api/models/status")
def models_status():
    return {"models": _models_status()}


@app.get("/api/vehicles/epa")
def search_epa_vehicles(
    search: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=20, ge=1, le=50),
):
    if not EPA_DATA.is_file():
        raise HTTPException(
            status_code=503,
            detail="The processed EPA vehicle catalogue is unavailable.",
        )
    try:
        vehicles = pd.read_csv(EPA_DATA)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read EPA vehicle catalogue: {exc}",
        ) from exc

    required = {
        "source_record_id",
        "year",
        "make",
        "model",
        *EPA_FEATURE_ORDER[1:],
    }
    missing = sorted(required - set(vehicles.columns))
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"EPA vehicle catalogue is missing required columns: {missing}",
        )

    query = search.strip().casefold()
    vehicles = vehicles.dropna(
        subset=[
            *EPA_FEATURE_ORDER,
            "source_record_id",
            "make",
            "model",
        ]
    )
    searchable = (
        vehicles["year"].astype("int64").astype(str)
        + " "
        + vehicles["make"].fillna("").astype(str)
        + " "
        + vehicles["model"].fillna("").astype(str)
        + " "
        + vehicles["vehicle_class"].fillna("").astype(str)
    ).str.casefold()
    matches = vehicles.loc[searchable.str.contains(query, regex=False)].head(limit)
    results = []
    for record in matches.to_dict(orient="records"):
        features = {}
        for name in EPA_FEATURE_ORDER:
            value = record[name]
            if pd.isna(value):
                features[name] = None
            elif name == "year":
                features[name] = int(value)
            elif name in {"drive", "vehicle_class"}:
                features[name] = str(value)
            else:
                features[name] = float(value)
        make = str(record["make"])
        model_name = str(record["model"])
        year = int(record["year"])
        results.append({
            "source_record_id": int(record["source_record_id"]),
            "label": f"{year} {make} {model_name}",
            "make": make,
            "model": model_name,
            "features": features,
        })
    return {
        "source": "EPA FuelEconomy.gov standardized vehicle/configuration records",
        "target": "EPA combined all-electric range estimate",
        "target_units": "miles",
        "results": results,
    }


def _epa_features_for_record(source_record_id: int) -> dict[str, Any]:
    if not EPA_DATA.is_file():
        raise HTTPException(
            status_code=503,
            detail="The processed EPA vehicle catalogue is unavailable.",
        )
    try:
        vehicles = pd.read_csv(EPA_DATA)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read EPA vehicle catalogue: {exc}",
        ) from exc
    if "source_record_id" not in vehicles.columns:
        raise HTTPException(
            status_code=500,
            detail="EPA vehicle catalogue is missing source_record_id.",
        )
    matches = vehicles.loc[
        pd.to_numeric(vehicles["source_record_id"], errors="coerce")
        == source_record_id
    ]
    if len(matches) != 1:
        raise HTTPException(
            status_code=404,
            detail="The selected EPA vehicle record was not found uniquely.",
        )
    record = matches.iloc[0]
    required = set(EPA_FEATURE_ORDER)
    missing = sorted(required - set(vehicles.columns))
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"EPA vehicle catalogue is missing required columns: {missing}",
        )
    if record[list(required)].isna().any():
        raise HTTPException(
            status_code=422,
            detail="The selected EPA vehicle record has incomplete prediction features.",
        )
    return {
        name: (
            int(record[name])
            if name == "year"
            else str(record[name])
            if name in {"drive", "vehicle_class"}
            else float(record[name])
        )
        for name in EPA_FEATURE_ORDER
    }


def _predict_epa_range(features: dict[str, Any]) -> dict[str, Any]:
    artifact = _load_epa_artifact()
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    feature_frame = pd.DataFrame([features], columns=EPA_FEATURE_ORDER)
    try:
        prediction = model.predict(preprocessor.transform(feature_frame))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"EPA range inference failed: {exc}",
        ) from exc
    predicted_miles = float(np.asarray(prediction).reshape(-1)[0])
    if not np.isfinite(predicted_miles) or predicted_miles <= 0:
        raise HTTPException(
            status_code=500,
            detail="EPA range model returned a non-finite or non-positive prediction.",
        )
    return {
        "predicted_epa_range_miles": round(predicted_miles, 2),
        "predicted_epa_range_km": round(predicted_miles * 1.609344, 2),
        "model": "XGBoost",
        "target_definition": artifact["target_definition"],
        "is_ml_prediction": True,
        "range_type": "standardized EPA vehicle/configuration estimate",
        "caveat": (
            "Not a real-world trip or remaining-range prediction; no weather "
            "or current vehicle state is modeled."
        ),
    }


@app.post("/api/route")
def calculate_route(req: RouteRequest):
    clean_start = req.start_address.strip()
    clean_dest = req.destination_address.strip()
    if len(clean_start) < 2 or len(clean_dest) < 2:
        raise HTTPException(
            status_code=422,
            detail="Starting point and destination must be at least 2 characters.",
        )
    if clean_start.lower() == clean_dest.lower():
        raise HTTPException(
            status_code=422,
            detail="Choose different starting and destination locations.",
        )

    def geocode(address: str) -> tuple[float, float, str]:
        clean_addr = address.strip()

        def _lookup(query: str):
            res = httpx.get(
                NOMINATIM_URL,
                params={"q": query, "format": "jsonv2", "limit": 1},
                headers={
                    "Accept": "application/json",
                    "Accept-Language": "en",
                    "User-Agent": "VoltWise-EV-research-prototype/1.0",
                },
                timeout=10.0,
            )
            res.raise_for_status()
            return res.json()

        try:
            results = _lookup(clean_addr)
            if (
                (not isinstance(results, list) or not results)
                and "vnr" in clean_addr.lower()
                and "jyothi" in clean_addr.lower()
            ):
                results = _lookup("VNR VJIET, Hyderabad")
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Address lookup service failed or timed out.",
            ) from exc
        if not isinstance(results, list) or not results:
            raise HTTPException(
                status_code=422,
                detail=f"Address not found: {clean_addr}",
            )
        item = results[0]
        try:
            latitude = float(item["lat"])
            longitude = float(item["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Address lookup service returned invalid coordinates.",
            ) from exc
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
        ):
            raise HTTPException(
                status_code=502,
                detail="Address lookup service returned out-of-range coordinates.",
            )
        return latitude, longitude, str(item.get("display_name") or address.strip())

    start = geocode(req.start_address)
    destination = geocode(req.destination_address)
    coordinates = f"{start[1]},{start[0]};{destination[1]},{destination[0]}"
    try:
        response = httpx.get(
            f"{OSRM_URL}/{coordinates}",
            params={"overview": "full", "geometries": "geojson", "steps": "false"},
            headers={"Accept": "application/json"},
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Road routing service failed or timed out.",
        ) from exc
    routes = payload.get("routes") if isinstance(payload, dict) else None
    route_code = payload.get("code") if isinstance(payload, dict) else None
    if route_code != "Ok" or not isinstance(routes, list) or not routes:
        raise HTTPException(
            status_code=422,
            detail="No drivable route was returned for these locations.",
        )
    route = routes[0]
    geometry = route.get("geometry", {}).get("coordinates")
    try:
        distance_m = float(route["distance"])
        duration_s = float(route["duration"])
        points = [
            {"latitude": float(pair[1]), "longitude": float(pair[0])}
            for pair in geometry
        ]
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Road routing service returned incomplete route data.",
        ) from exc
    if (
        len(points) < 2
        or not math.isfinite(distance_m)
        or not math.isfinite(duration_s)
        or distance_m < 0
        or duration_s < 0
        or any(
            not math.isfinite(point["latitude"])
            or not math.isfinite(point["longitude"])
            or not -90 <= point["latitude"] <= 90
            or not -180 <= point["longitude"] <= 180
            for point in points
        )
    ):
        raise HTTPException(
            status_code=502,
            detail="Road routing service returned invalid route measurements.",
        )
    return {
        "source": "OpenStreetMap Nominatim and OSRM",
        "start": {
            "latitude": start[0],
            "longitude": start[1],
            "label": start[2],
        },
        "destination": {
            "latitude": destination[0],
            "longitude": destination[1],
            "label": destination[2],
        },
        "distance_m": distance_m,
        "duration_s": duration_s,
        "geometry": points,
    }


@app.get("/api/weather/current")
def current_weather(
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=180),
):
    def unavailable(
        code: str,
        message: str,
        provider_error: str | None = None,
        provider_status_code: int | None = None,
    ) -> JSONResponse:
        error: dict[str, Any] = {"code": code, "message": message}
        if provider_error is not None:
            error["provider_error"] = provider_error
        if provider_status_code is not None:
            error["provider_status_code"] = provider_status_code
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "source": "Open-Meteo current conditions",
                "is_live_observation": False,
                "error": error,
            },
        )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        ),
        "timezone": "auto",
    }
    try:
        response = httpx.get(OPEN_METEO_URL, params=params, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return unavailable(
            "provider_http_error",
            "The weather provider rejected the current-conditions request.",
            provider_error=str(exc),
            provider_status_code=exc.response.status_code,
        )
    except httpx.RequestError as exc:
        return unavailable(
            "provider_request_failed",
            "The weather provider could not be reached.",
            provider_error=str(exc),
        )

    try:
        payload = response.json()
    except ValueError as exc:
        return unavailable(
            "invalid_provider_response",
            "The weather provider returned a response that could not be decoded.",
            provider_error=str(exc),
        )

    current = payload.get("current") if isinstance(payload, dict) else None
    required = (
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
        "time",
    )
    if not isinstance(current, dict) or any(key not in current for key in required):
        return unavailable(
            "incomplete_provider_response",
            "The weather provider response is missing required current conditions.",
        )
    try:
        values = {
            "temperature_c": float(current["temperature_2m"]),
            "relative_humidity_percent": float(
                current["relative_humidity_2m"]
            ),
            "wind_speed_kmh": float(current["wind_speed_10m"]),
            "precipitation_mm": float(current["precipitation"]),
        }
    except (TypeError, ValueError) as exc:
        return unavailable(
            "invalid_provider_measurements",
            "The weather provider returned non-numeric current conditions.",
            provider_error=str(exc),
        )
    if not all(np.isfinite(value) for value in values.values()):
        return unavailable(
            "invalid_provider_measurements",
            "The weather provider returned non-finite current conditions.",
        )
    return {
        **values,
        "observed_at": current["time"],
        "timezone": payload.get("timezone"),
        "source": "Open-Meteo current conditions",
        "is_live_observation": True,
    }


@app.post("/api/charging-stations/near-route")
def charging_stations_near_route(req: ChargingStationsRequest):
    cache_key = (
        req.radius_m,
        *sorted(
            (round(point.latitude, 3), round(point.longitude, 3))
            for point in req.points
        ),
    )
    now = time.monotonic()
    cached = _STATION_CACHE.get(cache_key)
    cache_hit = bool(cached and now - cached[0] < STATION_CACHE_TTL_SECONDS)
    is_stale_fallback = False
    if cache_hit:
        _, provider_stations, retrieved_at = cached
    else:
        if len(_STATION_CACHE) >= 100:
            oldest = min(_STATION_CACHE, key=lambda key: _STATION_CACHE[key][0])
            _STATION_CACHE.pop(oldest, None)

        # Optimize Overpass query:
        # Charging stations in OpenStreetMap are mapped on nodes or ways.
        # Omitting 'rel' prevents massive relation table scans that trigger HTTP 504 Gateway Timeouts.
        # If more than 3 points are supplied along a route, select up to 3 well-spaced points to prevent corridor overload.
        query_points = req.points
        if len(query_points) > 3:
            indices = [0, len(query_points) // 2, len(query_points) - 1]
            query_points = [query_points[i] for i in dict.fromkeys(indices)]

        clauses = " ".join(
            f"node[amenity=charging_station](around:{req.radius_m},{point.latitude},{point.longitude});"
            f"way[amenity=charging_station](around:{req.radius_m},{point.latitude},{point.longitude});"
            for point in query_points
        )
        query = f"[out:json][timeout:15];({clauses});out center tags 100;"
        payload = None
        last_error = None
        for attempt in range(OVERPASS_MAX_ATTEMPTS):
            try:
                response = httpx.post(
                    OVERPASS_URL,
                    data={"data": query},
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "VoltWise-EV-research-prototype/1.0",
                    },
                    timeout=OVERPASS_TIMEOUT_SECONDS,
                )
            except httpx.RequestError:
                last_error = "The station provider could not be reached or timed out."
            else:
                status_code = getattr(response, "status_code", 200)
                if status_code == 429 or status_code >= 500:
                    last_error = f"The station provider temporarily returned HTTP {status_code}."
                else:
                    try:
                        response.raise_for_status()
                        payload = response.json()
                    except (httpx.HTTPError, ValueError):
                        raise HTTPException(
                            status_code=502,
                            detail={"code": "invalid_provider_response", "message": "The station provider returned an unreadable response."},
                        )
                    break
            if attempt + 1 < OVERPASS_MAX_ATTEMPTS:
                time.sleep(OVERPASS_RETRY_BACKOFF_SECONDS * (2 ** attempt))

        if payload is None:
            # If live provider fails, check for previously cached data for this location
            stale_cached = _STATION_CACHE.get(cache_key)
            if stale_cached:
                _, provider_stations, retrieved_at = stale_cached
                is_stale_fallback = True
            else:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "provider_unavailable",
                        "message": last_error or "The station provider is temporarily unavailable.",
                        "retryable": True,
                    },
                )

        if not is_stale_fallback:
            elements = payload.get("elements") if isinstance(payload, dict) else None
            if not isinstance(elements, list):
                raise HTTPException(
                    status_code=502,
                    detail={"code": "invalid_provider_response", "message": "The station provider returned an invalid response."},
                )
            provider_stations = []
            for item in elements:
                if not isinstance(item, dict):
                    continue
                location = item
                if "lat" not in item or "lon" not in item:
                    location = item.get("center")
                if not isinstance(location, dict):
                    continue
                try:
                    latitude = float(location["lat"])
                    longitude = float(location["lon"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    not math.isfinite(latitude)
                    or not math.isfinite(longitude)
                    or not -90 <= latitude <= 90
                    or not -180 <= longitude <= 180
                ):
                    continue
                tags = item.get("tags")
                if not isinstance(tags, dict):
                    tags = {}
                connectors = {
                    key.removeprefix("socket:"): value
                    for key, value in tags.items()
                    if key.startswith("socket:") and value not in (None, "", "no")
                }
                provider_stations.append({
                    "id": item.get("id"),
                    "osm_type": item.get("type"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "name": tags.get("name"),
                    "operator": tags.get("operator") or tags.get("brand"),
                    "connectors": connectors,
                    "charging_power": tags.get("charging_station:output"),
                    "access": tags.get("access"),
                    "opening_hours": tags.get("opening_hours"),
                    "fee": tags.get("fee"),
                    "map_url": (
                        "https://www.openstreetmap.org/"
                        f"?mlat={latitude}&mlon={longitude}"
                        f"#map=17/{latitude}/{longitude}"
                    ),
                })
            retrieved_at = datetime.now(timezone.utc).isoformat()
            _STATION_CACHE[cache_key] = (time.monotonic(), provider_stations, retrieved_at)

    route = [
        (point.latitude, point.longitude)
        for point in (req.route_points or req.points)
    ]
    start = (req.points[0].latitude, req.points[0].longitude)
    end = (req.points[-1].latitude, req.points[-1].longitude)

    def distance_to_route(latitude: float, longitude: float) -> tuple[float, float]:
        if len(route) == 1:
            return (
                _haversine_m(latitude, longitude, route[0][0], route[0][1]),
                0.0,
            )
        best = math.inf
        best_progress = 0.0
        progress = 0.0
        cosine = math.cos(math.radians(latitude))
        station_x = longitude * 111320.0 * cosine
        station_y = latitude * 110540.0
        for first, second in zip(route, route[1:]):
            ax = first[1] * 111320.0 * cosine
            ay = first[0] * 110540.0
            bx = second[1] * 111320.0 * cosine
            by = second[0] * 110540.0
            dx, dy = bx - ax, by - ay
            denominator = dx * dx + dy * dy
            fraction = (
                max(
                    0.0,
                    min(
                        1.0,
                        ((station_x - ax) * dx + (station_y - ay) * dy)
                        / denominator,
                    ),
                )
                if denominator
                else 0.0
            )
            distance = math.hypot(
                station_x - (ax + fraction * dx),
                station_y - (ay + fraction * dy),
            )
            segment_length = _haversine_m(
                first[0], first[1], second[0], second[1]
            )
            if distance < best:
                best = distance
                best_progress = progress + fraction * segment_length
            progress += segment_length
        return best, best_progress

    stations = []
    for station in provider_stations:
        station = dict(station)
        route_distance, route_progress = distance_to_route(
            station["latitude"], station["longitude"]
        )
        station["distance_from_route_m"] = round(route_distance)
        station["route_progress_m"] = round(route_progress)
        station["distance_from_start_m"] = round(
            _haversine_m(
                station["latitude"], station["longitude"], start[0], start[1]
            )
        )
        station["distance_from_destination_m"] = round(
            _haversine_m(
                station["latitude"], station["longitude"], end[0], end[1]
            )
        )
        stations.append(station)
    stations.sort(key=lambda station: station["distance_from_route_m"])
    is_cached = cache_hit or is_stale_fallback
    return {
        "source": (
            "OpenStreetMap via Overpass"
            if not is_stale_fallback
            else "OpenStreetMap via Overpass (cached fallback)"
        ),
        "data_status": "cached" if is_cached else "live",
        "cached": is_cached,
        "retrieved_at": retrieved_at,
        "availability_status": "not_provided_by_source",
        "distance_method": "approximate nearest routed-polyline segment",
        "notice": (
            "Live station provider refresh timed out; displaying previously cached records."
            if is_stale_fallback
            else None
        ),
        "results": stations,
    }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6371000.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * earth_radius_m * math.asin(min(1.0, math.sqrt(value)))

@app.post("/api/predict/range")
def predict_range(req: RangeRequest):
    usable_kwh = req.battery_capacity_kwh * req.soc_percent / 100 * req.soh_percent / 100
    raw = usable_kwh * 1000 / req.consumption_wh_km
    return {"estimated_range_km": round(raw * req.weather_factor, 2), "unadjusted_range_km": round(raw, 2), "usable_energy_kwh": round(usable_kwh, 3), "method": "physics-based energy estimate", "weather_factor": req.weather_factor, "is_ml_prediction": False, "caveat": "Planning estimate only; not a validated real-world range model."}

@app.post("/api/predict/epa-range")
def predict_epa_range(req: EPARangeRequest):
    features = {
        "year": req.year,
        "city_e_kwh_100_miles": req.city_e_kwh_100_miles,
        "highway_e_kwh_100_miles": req.highway_e_kwh_100_miles,
        "combined_e_kwh_100_miles": req.combined_e_kwh_100_miles,
        "charge_240_hours": req.charge_240_hours,
        "drive": req.drive,
        "vehicle_class": req.vehicle_class,
    }
    return _predict_epa_range(features)


@app.post("/api/predict/epa-range/vehicle")
def predict_epa_vehicle_range(req: EPAVehicleRangeRequest):
    return _predict_epa_range(_epa_features_for_record(req.source_record_id))

def _load_nasa_retention_model():
    try:
        config = json.loads(NASA_RESEARCH_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail={"code": "nasa_registry_unavailable"}) from exc
    if (
        not isinstance(config, dict)
        or config.get("config_id") != "nasa_fy08q4:nasa_first_positive:xgboost"
        or config.get("feature_order") != NASA_FEATURE_ORDER
        or config.get("target", {}).get("name") != "first_positive_measurement_capacity_retention_percent"
        or config.get("input_constraints") != {
            "type": "integer",
            "minimum": 2,
            "maximum": 614,
            "range_basis": "Observed training-cell feature range (B0005 and B0018), 300 training observations.",
        }
        or config.get("preprocessing", {}).get("kind") != "median_impute_then_standardize"
    ):
        raise HTTPException(status_code=503, detail={"code": "nasa_registry_invalid"})
    artifact_name = config.get("model", {}).get("artifact_path")
    if artifact_name != "models/nasa_first_positive_retention_xgboost.json":
        raise HTTPException(status_code=503, detail={"code": "nasa_artifact_path_invalid"})
    artifact = (ROOT / artifact_name).resolve()
    if not artifact.is_relative_to(ROOT) or not artifact.is_file():
        raise HTTPException(status_code=503, detail={"code": "nasa_model_artifact_unavailable"})
    try:
        import numpy as np
        from xgboost import XGBRegressor
        preprocessing = config["preprocessing"]
        arrays = [np.asarray(preprocessing[key], dtype=np.float64) for key in ("imputer_statistics", "mean", "scale")]
        if any(arr.shape != (len(NASA_FEATURE_ORDER),) or not np.isfinite(arr).all() for arr in arrays):
            raise ValueError("Invalid preprocessing dimensions or values")
        if np.any(arrays[2] == 0):
            raise ValueError("Invalid zero preprocessing scale")
        estimator = XGBRegressor()
        estimator.load_model(str(artifact))
        if estimator.get_booster().num_features() != len(NASA_FEATURE_ORDER):
            raise ValueError("Model feature count does not match registry")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "nasa_model_load_failed", "reason": type(exc).__name__},
        ) from exc
    return config, estimator


@app.post("/api/predict/nasa-capacity-retention")
def predict_nasa_capacity_retention(req: NasaRetentionRequest):
    config, estimator = _load_nasa_retention_model()
    import numpy as np
    import pandas as pd

    values = np.asarray([[req.cycle_index_in_file]], dtype=np.float64)
    preprocessing = config["preprocessing"]
    imputer = np.asarray(preprocessing["imputer_statistics"], dtype=np.float64)
    mean = np.asarray(preprocessing["mean"], dtype=np.float64)
    scale = np.asarray(preprocessing["scale"], dtype=np.float64)
    values = np.where(np.isfinite(values), values, imputer)
    values = (values - mean) / scale
    try:
        prediction = float(estimator.predict(pd.DataFrame(values, columns=config["feature_order"]))[0])
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "nasa_inference_failed", "reason": type(exc).__name__},
        ) from exc
    if not math.isfinite(prediction):
        raise HTTPException(status_code=500, detail={"code": "nasa_prediction_not_finite"})
    return {
        "status": "prediction",
        "config_id": config["config_id"],
        "dataset": config["dataset"],
        "model": config["model"]["name"],
        "input_features": config["feature_order"],
        "input_values": {"cycle_index_in_file": req.cycle_index_in_file},
        "target": config["target"]["name"],
        "target_label": config["target"]["label"],
        "target_definition": config["target"]["definition"],
        "value": prediction,
        "unit": config["target"]["unit"],
        "provisional_proxy": True,
        "personalized_to_individual_battery": False,
        "notice": config["proxy_notice"],
        "limitations": config["limitations"],
    }


@app.post("/api/predict/soh")
def predict_soh(req: SOHRequest):
    artifact_path = MODELS / "battery_soh_model.joblib"
    if not artifact_path.is_file():
        raise HTTPException(status_code=503, detail=SOH_UNAVAILABLE_REASON)
    try:
        from app.soh_service import SOHPredictionRequest, run_soh_prediction
        pred_req = SOHPredictionRequest(
            battery_id=req.battery_id or "unknown",
            cycle_index=max(1.0, float(req.cycle_index)),
            voltage_mean=float(req.voltage_mean),
            current_mean=float(req.current_mean),
            temperature_mean=float(req.temperature_mean),
            capacity_ah=req.capacity_ah,
            rated_capacity_ah=getattr(req, "rated_capacity_ah", 2.0) or 2.0,
        )
        return run_soh_prediction(pred_req, MODELS)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=SOH_UNAVAILABLE_REASON) from exc


@app.post("/api/predict/battery-soh")
def predict_battery_soh(req: SOHRequest):
    return predict_soh(req)


@app.get("/api/soh/metrics")
def get_soh_metrics():
    metrics_path = RESULTS / "battery_soh_metrics.json"
    if not metrics_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Battery SOH evaluation metrics file not found.",
        )
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read SOH evaluation metrics: {exc}",
        ) from exc


@app.post("/api/predict/trip-distance")
def predict_trip_distance(req: TripDistancePredictRequest):
    model_name = req.model_name.strip()
    norm = model_name.lower().replace("-", "_").replace(" ", "_")

    if "base" in norm:
        mean_km = 2.601
        metrics_p = RESULTS / "trip_distance_metrics.json"
        if metrics_p.exists():
            try:
                m_data = json.loads(metrics_p.read_text(encoding="utf-8"))
                for exp in m_data.get("experiments", []):
                    if "base" in exp.get("model", "").lower() and "train_mean_km" in exp:
                        mean_km = float(exp["train_mean_km"])
                        break
            except Exception:
                pass
        return {
            "task": "VED Initial-Window EV Trip Distance Prediction",
            "predicted_trip_distance_km": round(mean_km, 3),
            "model": "Training-Mean Baseline",
            "target_units": "km",
            "observation_window": "30-second initial telemetry window (1 Hz)",
            "is_research_estimate": True,
            "dataset": "Vehicle Energy Dataset (2013 Nissan Leaf pure-EV trips)",
            "notice": (
                "Research estimate of total completed trip distance in km, derived from "
                "an initial 30-second observation window. This is NOT remaining driving range "
                "and is not physical battery state-of-health."
            ),
            "limitations": [
                "Baseline simply predicts the training-set mean completed trip distance.",
                "Dataset contains only three 2013 Nissan Leafs; cannot claim generalization across vehicle types.",
            ],
        }

    elif "xgb" in norm:
        xgb_path = MODELS / "trip_distance_xgboost.joblib"
        if not xgb_path.is_file():
            raise HTTPException(status_code=503, detail="Trip distance XGBoost model artifact is not installed.")
        try:
            artifact = joblib.load(xgb_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load XGBoost artifact: {exc}") from exc

        feature_dict = {
            "window_duration_s": 30.0,
            "window_distance_km": req.window_distance_km,
            "speed_mean": req.speed_mean,
            "speed_max": req.speed_max,
            "speed_min": req.speed_min,
            "speed_std": req.speed_std,
            "power_mean": req.power_mean,
            "power_max": req.power_max,
            "power_min": req.power_min,
            "current_mean": req.current_mean,
            "voltage_mean": req.voltage_mean,
            "soc_start": req.soc_start,
            "soc_current": req.soc_current,
            "soc_delta": req.soc_delta,
            "temperature_mean": req.temperature_mean,
        }
        features = np.asarray([[feature_dict[name] for name in TRIP_DISTANCE_TABULAR_FEATURES]], dtype=np.float64)
        pred = float(artifact["model"].predict(features)[0])
        if not math.isfinite(pred):
            raise HTTPException(status_code=500, detail="XGBoost prediction returned non-finite value.")

        return {
            "task": "VED Initial-Window EV Trip Distance Prediction",
            "predicted_trip_distance_km": round(max(0.0, pred), 3),
            "model": "XGBoost",
            "target_units": "km",
            "observation_window": "30-second initial telemetry window (1 Hz)",
            "is_research_estimate": True,
            "dataset": "Vehicle Energy Dataset (2013 Nissan Leaf pure-EV trips)",
            "notice": (
                "Research estimate of total completed trip distance in km, derived from "
                "an initial 30-second observation window. This is NOT remaining driving range "
                "and is not physical battery state-of-health."
            ),
            "limitations": [
                "Dataset contains only three 2013 Nissan Leafs; cannot claim generalization across vehicle types.",
                "Early 30s driving patterns cannot predict unexpected route stops or traffic variations.",
            ],
        }

    elif any(k in norm for k in ["lstm", "gru", "hybrid", "cnn"]):
        import torch
        from training.train_trip_distance_experiment import LSTMModel, GRUModel, CNNLSTMModel

        if "hybrid" in norm or "cnn" in norm:
            model_key = "cnn_lstm"
            model_label = "Hybrid CNN-LSTM"
            model_cls = CNNLSTMModel
        elif "gru" in norm:
            model_key = "gru"
            model_label = "GRU"
            model_cls = GRUModel
        else:
            model_key = "lstm"
            model_label = "LSTM"
            model_cls = LSTMModel

        pt_path = MODELS / f"trip_distance_{model_key}.pt"
        if not pt_path.is_file():
            raise HTTPException(status_code=503, detail=f"Trip distance {model_label} model artifact is not installed.")

        checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
        scaler = checkpoint["scaler"]

        if req.sequence is not None:
            seq_arr = np.asarray(req.sequence, dtype=np.float32)
            if seq_arr.shape != (30, 7):
                raise HTTPException(status_code=422, detail=f"Sequence must have shape (30, 7); got {seq_arr.shape}")
        else:
            rows = []
            step_km = (req.speed_mean / 3600.0)
            for step_i in range(30):
                cum_dist = req.window_distance_km * (step_i / 29.0) if req.window_distance_km > 0 else step_i * step_km
                rows.append([
                    req.speed_mean,
                    req.current_mean,
                    req.voltage_mean,
                    req.power_mean,
                    req.soc_start + (req.soc_delta * (step_i / 29.0)),
                    req.temperature_mean,
                    cum_dist,
                ])
            seq_arr = np.asarray(rows, dtype=np.float32)

        scaled_seq = scaler.transform(seq_arr).reshape(1, 30, 7)
        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32)

        net = model_cls(input_dim=7, hidden_dim=32)
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        with torch.no_grad():
            pred = float(net(input_tensor).item())

        if not math.isfinite(pred):
            raise HTTPException(status_code=500, detail=f"{model_label} prediction returned non-finite value.")

        return {
            "task": "VED Initial-Window EV Trip Distance Prediction",
            "predicted_trip_distance_km": round(max(0.0, pred), 3),
            "model": model_label,
            "target_units": "km",
            "observation_window": "30-second initial telemetry window (1 Hz)",
            "is_research_estimate": True,
            "dataset": "Vehicle Energy Dataset (2013 Nissan Leaf pure-EV trips)",
            "notice": (
                "Research estimate of total completed trip distance in km, derived from "
                "an initial 30-second observation window. This is NOT remaining driving range "
                "and is not physical battery state-of-health."
            ),
            "limitations": [
                "Dataset contains only three 2013 Nissan Leafs; cannot claim generalization across vehicle types.",
                "Sequence model was trained with a vehicle-held-out validation checkpoint on limited trips.",
            ],
        }

    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported model: {model_name}. Supported models: XGBoost, LSTM, GRU, Hybrid CNN-LSTM, Training-Mean Baseline.",
        )


@app.get("/api/model-performance")
def model_performance():
    p = RESULTS / "model_metrics.json"
    experiments = []
    response = {
        "dataset_status": dataset_status()["status"],
        "experiments": experiments,
    }
    try:
        nasa_config = json.loads(NASA_RESEARCH_REGISTRY.read_text(encoding="utf-8"))
        response["nasa_capacity_retention"] = {
            "config_id": nasa_config["config_id"],
            "dataset": nasa_config["dataset"],
            "target": nasa_config["target"],
            "evaluation": nasa_config["evaluation"],
            "proxy_notice": nasa_config["proxy_notice"],
        }
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise HTTPException(status_code=500, detail="NASA research metrics metadata is unavailable or invalid.") from exc
    if p.exists():
        try:
            general = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read metrics file: {exc}",
            ) from exc
        if isinstance(general, dict):
            experiments.extend(general.get("experiments", []))
            response.update({
                key: value
                for key, value in general.items()
                if key not in {"experiments", "dataset_status"}
            })
    epa_path = RESULTS / "epa_range_metrics.json"
    if epa_path.exists():
        try:
            epa = json.loads(epa_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read EPA range metrics: {exc}",
            ) from exc
        if isinstance(epa, dict):
            experiments.extend(epa.get("experiments", []))
            response["epa_range_status"] = epa.get("dataset_status")
            response["epa_range_experiments"] = epa.get(
                "non_evaluated_models", []
            )

    trip_dist_path = RESULTS / "trip_distance_metrics.json"
    if trip_dist_path.exists():
        try:
            trip_data = json.loads(trip_dist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read trip distance metrics: {exc}",
            ) from exc
        if isinstance(trip_data, dict):
            experiments.extend(trip_data.get("experiments", []))
            response["trip_distance_research"] = trip_data

    if not experiments:
        response["message"] = (
            "No evaluated benchmark results found. Untrained models and "
            "unsupported tasks are not reported as completed."
        )
    return response

@app.get("/api/datasets/status")
def dataset_status():
    p = ROOT / "data" / "processed" / "battery_measurements.csv"
    if not MANIFEST.exists():
        return {
            "status": "not_configured",
            "harmonized_battery_csv": p.exists(),
            "path": str(p.relative_to(ROOT)),
            "message": f"Dataset manifest not found: {MANIFEST}",
        }
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read dataset manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=500, detail="Dataset manifest root must be a JSON object.")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise HTTPException(status_code=500, detail="Dataset manifest must contain a datasets array.")
    nasa = next(
        (
            item
            for item in datasets
            if isinstance(item, dict)
            and item.get("source", "").startswith("NASA")
        ),
        None,
    )
    return {
        "status": nasa.get("status", "unknown") if nasa else "not_configured",
        "harmonized_battery_csv": p.exists(),
        "path": str(p.relative_to(ROOT)),
        "sample_count": nasa.get("sample_count") if nasa else None,
        "battery_count": nasa.get("battery_count") if nasa else None,
        "excluded_observation_count": nasa.get("excluded_observation_count") if nasa else None,
        "review_flagged_row_count": nasa.get("review_flagged_row_count") if nasa else None,
        "soh_target_status": (
            nasa.get("calculated_targets", {})
            .get("soh_percent", {})
            .get("status")
            if nasa else None
        ),
        "sources": [
            {
                "source": item.get("source"),
                "status": item.get("status"),
                "sample_count": item.get(
                    "sample_count",
                    item.get("valid_record_count"),
                ),
            }
            for item in datasets
        ],
        "range_target_column": (
            "epa_range_miles (standardized EPA vehicle/configuration estimate; "
            "not real-world trip range)"
        ),
    }
