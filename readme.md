# Urban Planning Simulation Tool 

## Summary
Complete end-to-end MLOps system for flood risk simulation with the following components:

| Component | Tech | 
|-----------|------|
| **API Backend** | FastAPI + SQLAlchemy |
| **Frontend** | Next.js + TypeScript + Lucide React |
| **MLOps Pipeline** | MLflow + joblib |
| **Containerization** | Docker Compose |
| **CI/CD** | GitHub Actions |
| **Tests** | pytest + TestClient |

---

## Project Structure

```
ml-opsidian-genesis-initial-round-26/
├── api/                          # FastAPI Backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI app (5 endpoints)
│   ├── schemas.py                # Pydantic validation (30+ fields)
│   ├── feature_engine.py         # 115-feature pipeline
│   ├── model_manager.py          # Model loader + mock mode
│   ├── database.py               # PostgreSQL + SQLAlchemy
│   ├── logger.py                 # Prediction logging
│   ├── requirements.txt
│   └── tests/
│       ├── __init__.py
│       ├── test_schemas.py       # 10 validation tests
│       ├── test_api.py           # 10 integration tests
│       └── test_feature_engine.py # 14 feature/risk tests
├── frontend/                     # Next.js Frontend
│   ├── package.json              # + lucide-react
│   ├── tsconfig.json
│   ├── next.config.js
│   └── src/
│       ├── app/
│       │   ├── globals.css       # 800+ line design system (#eb002d primary)
│       │   ├── layout.tsx        # Root layout
│       │   ├── page.tsx          # Simulation page (Lucide icons)
│       │   └── analytics/
│       │       └── page.tsx      # MLOps dashboard (Lucide icons)
│       └── components/
│           ├── Navbar.tsx        # Waves, Zap, BarChart3 icons
│           ├── FeatureSlider.tsx
│           ├── RiskGauge.tsx
│           └── FeatureImportanceChart.tsx
├── mlops/                        # Model Export
│   ├── __init__.py               # MLflow config
│   ├── export_v13.py             # V13: 10-model Ridge stack
│   ├── export_v18.py             # V18 Titan: 36-model HuberRegressor
│   └── export_v20.py             # V20 Colossus: 270-model HuberRegressor
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml            # 4 services
└── .github/workflows/ci.yml     # CI pipeline
```

---

## Model Export Scripts

| Script | Models | Meta-Model | Checkpoint Dir | Output Dir |
|--------|--------|-----------|----------------|------------|
| export_v13.py | 10 (LGB + CatBoost) | Ridge | N/A (trains from scratch) | `mlops/artifacts/v13/` |
| export_v18.py | 36 (HGB + LGB + XGB + CatBoost) | HuberRegressor(ε=1.35) | `titan_checkpoints/` | `mlops/artifacts/v18/` |
| export_v20.py | ~270 (same families, 10 seeds) | HuberRegressor(ε=1.35) | `colossus_checkpoints/` | `mlops/artifacts/v20/` |

Each export script produces 6 artifact files:
- `pipeline.joblib` — Serialized inference pipeline (base models + meta-model)
- `label_encoders.json` — Category → integer mappings
- `te_stats.json` — Target encoding statistics per categorical column
- `feature_columns.json` — Ordered feature names
- `medians.json` — Training medians for NaN imputation
- `global_mean.json` — Target mean (fallback for unknown TE keys)

---

## How to Run

### Option 1: Docker Compose (Recommended)
```bash
docker-compose up --build
```
This starts 4 services:
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000 (Swagger docs at /docs)
- **MLflow**: http://localhost:5000
- **PostgreSQL**: localhost:5432

### Option 2: Manual (Development)

**Backend:**
```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Export Models (Optional — mock mode works without them):**
```bash
python -m mlops.export_v13    # ~30 min
python -m mlops.export_v18    # ~2 hrs (needs titan_checkpoints/)
python -m mlops.export_v20    # ~8 hrs (needs colossus_checkpoints/)
```

### Run Tests
```bash
pip install pytest httpx
python -m pytest api/tests/ -v
```
