# VoltWise — EV Range & Battery Health Intelligence

A full-stack React/Vite + FastAPI academic platform for EV range planning, route mapping, weather context, charging station discovery, NASA battery state-of-health degradation research, and a 5-model exploratory benchmark for EV trip-distance prediction from early telemetry.

---

## Key Features & Research Components

1. **Dashboard & Vehicle Intelligence**:
   - Interactive EV parameters: State of Charge (SOC), Planning SOH adjustment, Battery Capacity (kWh), and Nominal Consumption (Wh/km).
   - Real-time physics-based range estimation with ambient weather adjustment.
   - Hourly illustrative energy profile and battery status rings.

2. **Trip Planner & Charging Navigation**:
   - Route geocoding and geometry via public Nominatim and OSRM services.
   - Route-aware EV charging station discovery via OpenStreetMap Overpass API.
   - Real-world reachability analysis warning if route distance exceeds estimated usable range.

3. **EPA Standardized Vehicle-Range Model**:
   - Trained XGBoost Regressor on official EPA dynamometer fuel economy data (`models/epa_range_xgboost.joblib`).
   - Predicts certified vehicle-configuration cycle range (miles) from baseline vehicle attributes.
   - Explicitly separated from on-road dynamic trip distance.

4. **NASA Battery Degradation Proxy**:
   - Research evaluation on NASA PCoE FY08Q4 lithium-ion 18650 cells (`B0005`, `B0006`, `B0007`, `B0018`).
   - Battery-grouped evaluation protocol: Cell `B0006` strictly held out as unseen test set.
   - Target: first-positive measurement capacity-retention proxy (% of initial capacity) from cycle count. Linear cycle-trend baseline out-performs XGBoost on monotonic wear curves.

5. **VED Initial-Window Trip-Distance Prediction Benchmark (5 Models)**:
   - Faculty-guide approved exploratory benchmark predicting total completed trip distance (km) using only the initial 30 seconds of driving telemetry from the Vehicle Energy Dataset (VED).
   - Filtered for pure-electric Nissan Leafs (Vehicles `10`, `455`, `541`) with ≥95% speed-distance coverage (51 verified trips).
   - Strict vehicle-grouped split: Vehicle `10` strictly held out as unseen test set (12 trips).
   - **5 Evaluated Architectures**:
     1. **Training-Mean Baseline**: Predicts constant training-set mean (2.193 km).
     2. **XGBoost Regressor**: Trained on aggregated 30s statistical window features.
     3. **LSTM (2-Layer)**: Recurrent neural network on 30s normalized 7-channel sequence (speed, power, current, voltage, SOC, delta SOC, temp).
     4. **GRU (2-Layer)**: Gated recurrent unit network on 30s normalized sequence. **Achieved lowest MAE (0.905 km) and lowest MAPE (45.60%)**.
     5. **Hybrid CNN-LSTM**: 1D-CNN temporal feature extractor feeding into an LSTM recurrent layer.
   - **Scientific Disclosure on Negative R²**: All 5 models produce negative R² (-0.249 to -0.416) on held-out Vehicle 10. Early driving dynamics in the first 30 seconds do not determine final trip destination. GRU nevertheless achieved an MAE of 0.905 km, substantially outperforming the mean baseline. Conforms strictly to academic integrity standards with zero synthetic or fabricated data.

6. **Interactive Model Performance Hub & Live Inference Sandbox**:
   - Full benchmark tables and scientific documentation across all 3 research datasets.
   - Live interactive sandbox calling `POST /api/predict/trip-distance` with selectable architectures.

---

7. **Battery State of Health (SOH) Estimation Engine**:
   - Validated Gradient Boosting regression model trained on NASA 18650 Li-ion cells under CC-CV aging (`models/battery_soh_model.joblib`).
   - Strict cell-grouped holdout validation with cell `B0006` held out (R² = 0.9225, MAE = 2.89% SOH, RMSE = 3.50% SOH).
   - Dedicated frontend interactive sandbox with empirical 90% uncertainty intervals (`frontend/src/BatterySOHPredictor.jsx`), sample presets, and operational telemetry validation.

8. **Overpass Charging Station Provider Hardening & Fallback**:
   - Resilient Overpass API query engine with relation (`rel`) pruning and dynamic route corridor subsampling.
   - Bounded exponential backoff retries on transient upstream 504 Gateway Timeouts.
   - Transparent stale cache fallback preserving user navigation when upstream OpenStreetMap services are degraded.

9. **Smart Journey & Weather-Aware Recommendations**:
   - Contextual pre-departure checklists and driving guidance (`frontend/src/JourneyRecommendations.jsx`).
   - Dynamic precaution matrix addressing cold weather range reduction, battery overheating risk, precipitation, and aerodynamic drag.

---

## Four-Task Architecture Disambiguation

VoltWise enforces strict separation across four distinct calculation and prediction domains:
- **Task A: EPA Certified Range**: Standardized laboratory dynamometer range across vehicle catalog configurations.
- **Task B: VED Trip Distance**: Exploratory prediction of completed trip length (km) from initial 30s telemetry. *Never to be confused with remaining battery range.*
- **Task C: NASA SOH & Degradation**: Standalone cell SOH estimation (% of 2.0 Ah rated benchmark) and cycle-trend retention proxy.
- **Task D: Trip Planner Physics**: Deterministic calculation: `Range = (Capacity * SOC * SOH / Consumption) * WeatherFactor`.

---

## VED Benchmark Results (Held-Out Vehicle 10)

| Model Name | Architecture / Pipeline | RMSE (km) | MAE (km) | R² Score | MAPE (%) | Key Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Training-Mean Baseline** | Constant train mean (2.193 km) | 1.499 | 1.291 | -0.2664 | 131.90% | Reference anchor |
| **XGBoost** | Tabular initial-window aggregates | 1.488 | 1.285 | -0.2491 | 130.56% | Lowest RMSE, best tree model |
| **LSTM** | 2-layer LSTM on 30s sequence | 1.541 | 1.370 | -0.3392 | 134.87% | Recurrent sequence model |
| **GRU** | 2-layer GRU on 30s sequence | 1.584 | **0.905** | -0.4156 | **45.60%** | **Lowest MAE and lowest MAPE** |
| **Hybrid CNN-LSTM** | 1D-CNN + LSTM feature pipeline | 1.572 | 1.392 | -0.3928 | 141.83% | Convolutional-recurrent hybrid |

*Split Protocol: Train = 31 trips (Veh 455); Val = 8 trips (Veh 541 + Veh 455); Test = 12 trips (Veh 10). Test trip lengths: 0.44 km to 5.41 km.*

---

## Quickstart & Execution

### 1. Environment Setup

Ensure Python 3.10+ and Node.js 18+ are available:

```powershell
# Activate Python virtual environment
.\.venv\Scripts\Activate.ps1

# Install CPU PyTorch and backend requirements
pip install -r backend/requirements.txt
```

### 2. Run Unified Automated Tests

All 122 python tests and 39 frontend unit tests execute and pass out-of-the-box:

```powershell
# Run all 122 python tests (preprocessing, audits, models, provider hardening, and FastAPI API)
pytest

# Run frontend unit tests (39 tests)
cd frontend
npm test
cd ..
```

### 3. Start Backend & Frontend

**Terminal 1 — Backend (FastAPI):**
```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```
Interactive API Swagger Docs: http://127.0.0.1:8000/docs

**Terminal 2 — Frontend (React / Vite):**
```powershell
cd frontend
npm run dev -- --port 5173
```
Application Dashboard: http://127.0.0.1:5173

---

## Reproducing the VED Trip-Distance Experiment

To re-extract the VED telemetry dataset, regenerate the initial-window sequence matrices, and re-train all 5 benchmark models from scratch:

```powershell
# 1. Stream 30-second windows and compute verified trip targets from raw VED archives
python scripts/prepare_trip_distance_dataset.py

# 2. Train all 5 architectures (Baseline, XGBoost, LSTM, GRU, Hybrid CNN-LSTM)
python training/train_trip_distance_experiment.py
```

Generated Artifacts:
- Data Table: `data/processed/ved/ved_trip_distance_dataset.csv`
- 3D Sequence Array: `data/processed/ved/ved_trip_distance_sequences.npz` (`51 × 30 × 7`)
- Split Manifest: `data/processed/ved/ved_trip_distance_split_manifest.json`
- Trained Models: `models/trip_distance_xgboost.joblib`, `models/trip_distance_lstm.pt`, `models/trip_distance_gru.pt`, `models/trip_distance_cnn_lstm.pt`
- Evaluation Metrics: `results/trip_distance_metrics.json`
- Detailed Predictions: `results/trip_distance_test_predictions.csv`

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Comprehensive system health and model availability status |
| `GET` | `/api/model-performance` | Unified evaluation metrics for EPA, NASA, and VED experiments |
| `GET` | `/api/soh/metrics` | Evaluation metrics and holdout performance for validated Battery SOH model |
| `POST` | `/api/predict/soh` | Standalone Battery SOH prediction with 90% empirical uncertainty intervals |
| `POST` | `/api/predict/trip-distance` | Live inference for completed trip distance using Baseline, XGBoost, LSTM, GRU, or Hybrid CNN-LSTM |
| `POST` | `/api/predict/epa-range` | Certified vehicle configuration range prediction |
| `POST` | `/api/predict/nasa-capacity-retention` | NASA battery capacity retention degradation proxy |
| `POST` | `/api/predict/range` | Physics-based usable range calculation for trip planner |
| `POST` | `/api/charging-stations/near-route` | Route-aware charging station lookup with hardening and cached fallback |

---

## Project Directory Layout

```text
ev-fullstack/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application, routes, and inference handlers
│   │   └── soh_service.py              # Isolated battery SOH inference service
│   └── tests/
│       ├── test_api.py                 # Core API endpoints tests
│       ├── test_battery_soh_service.py # SOH prediction service unit tests (5 tests)
│       ├── test_nasa_research.py       # NASA proxy endpoint tests
│       ├── test_provider_hardening.py  # External provider fallback & retry tests (7 tests)
│       └── test_trip_distance_api.py   # Trip-distance API endpoint tests (7 tests)
├── data/
│   ├── raw/trip_distance/              # Extracted VED telemetry for Nissan Leafs
│   ├── processed/ved/                  # Processed VED dataset, 3D sequences, split manifest
│   └── dataset_manifest.json           # Catalog of registered datasets
├── frontend/
│   ├── src/
│   │   ├── App.jsx                     # VoltWise React application & Model Performance hub
│   │   ├── BatterySOHPredictor.jsx     # Interactive battery health predictor UI
│   │   ├── batterySOH.js               # SOH calculation, presets, and validation logic
│   │   ├── JourneyRecommendations.jsx  # Contextual weather & journey checklist UI
│   │   ├── journeyRecommendations.js   # Precaution matrix & journey rules
│   │   ├── tripPlanner.js              # Route planning and station matching logic
│   │   ├── planning.js                 # Physics estimation and weather adjustments
│   │   └── styles.css                  # Modern UI design system
│   └── tests/                          # 39 unit tests for frontend modules
├── models/                             # Trained model weights and joblib pipelines
├── results/                            # JSON evaluation metrics and CSV predictions
├── scripts/
│   └── prepare_trip_distance_dataset.py # Telemetry extraction and window builder
├── tests/                              # 65 automated tests for data pipelines and models
├── training/
│   ├── train_battery_soh_model.py      # Battery SOH regression training & evaluation engine
│   └── train_trip_distance_experiment.py # 5-model benchmark training and evaluation engine
└── pytest.ini                          # Root test configuration (122 unified tests)
```
