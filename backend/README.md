# VoltWise FastAPI backend

## Run locally (Windows PowerShell)
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs.

- `GET /api/health`: backend/model availability
- `POST /api/predict/range`: transparent physics-based estimate
- `POST /api/predict/soh`: uses a metadata-bearing `models/xgboost_soh.joblib` (503 until a compatible trained artifact is installed)
- `GET /api/model-performance`: reads `results/model_metrics.json`, or reports that results are unavailable
- `GET /api/datasets/status`: reads `data/dataset_manifest.json` and reports source readiness

The supported SOH artifact declares target `soh_percent`, preprocessing `none`, and feature order `cycle_index`, `voltage_mean`, `current_mean`, `temperature_mean`. Measured `capacity_ah` is accepted for compatibility with the current form but is not passed to the model. Inference rejects artifacts without matching metadata and does not clip predictions.

No trained model or research result is bundled. The current NASA target is provisional, so the training script refuses to emit metrics until the label reference and protocol comparability have been reviewed. The range endpoint is a physics estimate, not an ML prediction.

Run backend tests from this directory:
```powershell
py -m pytest tests
```
