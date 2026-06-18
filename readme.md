# Urban Planning Simulation Tool

## Summary

Complete end-to-end MLOps system for flood risk simulation with the following components:

| Component            | Tech                                |
| -------------------- | ----------------------------------- |
| **API Backend**      | FastAPI + SQLAlchemy                |
| **Frontend**         | Next.js + TypeScript + Lucide React |
| **MLOps Pipeline**   | MLflow + joblib                     |
| **Containerization** | Docker Compose                      |
| **CI/CD**            | GitHub Actions                      |
| **Tests**            | pytest + TestClient                 |

---

<!-- ## AI Counterfactual Flood Mitigation Planner

The **AI Mitigation Planner** is a decision-support simulation feature for exploring how a limited set of planning interventions could change the model-predicted flood risk for a district. Users select a target risk level, budget profile, maximum number of actions, and allowed intervention types. The planner returns:

- baseline and optimized risk scores;
- whether the requested target was reached;
- an ordered mitigation plan with estimated model impact;
- the candidate search trace; and
- alternative plans when available.

These recommendations are counterfactual model scenarios, not guarantees of flood prevention or real-world project outcomes. They should be reviewed alongside engineering, hydrological, financial, environmental, and policy analysis.

### How it works

1. The planner starts with the existing default features for the selected district.
2. It generates bounded changes for drainage, infrastructure, emergency access, healthcare access, river setback, and built-up area.
3. Each candidate is scored through the existing `model_manager.predict(...)` inference path.
4. A deterministic greedy search selects the best budget-adjusted improving action at each step.
5. Search stops when the target threshold is reached, no candidate improves the score, or `max_steps` is exhausted.

The planner treats the current model as a black-box scorer. It **does not retrain, fine-tune, recalibrate, replace, or modify the model or its artifacts**.

### API

`POST /api/optimize-plan`

Example request:

```json
{
  "district": "Colombo",
  "model_version": "v13",
  "target_risk_level": "Medium",
  "max_steps": 3,
  "budget_profile": "balanced",
  "allowed_features": [
    "drainage_index",
    "infrastructure_score",
    "nearest_evac_km",
    "nearest_hospital_km",
    "distance_to_river_m",
    "built_up_percent"
  ]
}
```

Example response (illustrative scores):

```json
{
  "district": "Colombo",
  "model_version": "v13",
  "baseline_score": 0.6123,
  "baseline_risk_level": "High",
  "target_risk_level": "Medium",
  "target_score_threshold": 0.5,
  "target_reached": true,
  "optimized_score": 0.4817,
  "optimized_risk_level": "Medium",
  "risk_reduction_pct": 21.33,
  "recommended_plan": [
    {
      "feature": "drainage_index",
      "feature_label": "Drainage System",
      "category": "Infrastructure",
      "from_value": 0.38,
      "to_value": 0.85,
      "score_before": 0.6123,
      "score_after": 0.4817,
      "absolute_reduction": 0.1306,
      "relative_reduction_pct": 21.33,
      "cost_level": "low",
      "rationale": "Improve drainage capacity to move stormwater away from exposed areas."
    }
  ],
  "search_trace": [
    {
      "step": 1,
      "tried_feature": "drainage_index",
      "tried_value": 0.85,
      "resulting_score": 0.4817,
      "resulting_risk_level": "Medium",
      "accepted": true
    }
  ],
  "alternatives": [
    {
      "name": "Alternative via Infrastructure Score",
      "optimized_score": 0.4972,
      "optimized_risk_level": "Medium",
      "target_reached": true,
      "risk_reduction_pct": 18.8,
      "steps": []
    }
  ]
}
```

Supported budget profiles are `low_cost`, `balanced`, and `aggressive`. The optional `allowed_features` field may be omitted to search all supported interventions.

### Frontend

Open [http://localhost:3000/optimize](http://localhost:3000/optimize), or select **AI Planner** in the navbar.

-->

## Project Structure

```
ml-opsidian-genesis-initial-round-26/
├── api/                          # FastAPI Backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI app and REST endpoints
│   ├── schemas.py                # Pydantic validation (30+ fields)
│   ├── feature_engine.py         # 115-feature pipeline
│   ├── model_manager.py          # Model loader + mock mode
│   ├── counterfactual_planner.py # Black-box mitigation search
│   ├── database.py               # PostgreSQL + SQLAlchemy
│   ├── logger.py                 # Prediction logging
│   ├── requirements.txt
│   └── tests/
│       ├── __init__.py
│       ├── test_schemas.py       # 10 validation tests
│       ├── test_api.py           # 10 integration tests
│       ├── test_counterfactual_planner.py # Planner verification
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
│       │   ├── optimize/
│       │   │   └── page.tsx      # AI mitigation planner
│       │   └── analytics/
│       │       └── page.tsx      # MLOps dashboard (Lucide icons)
│       └── components/
│           ├── Navbar.tsx        # Application navigation
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

| Script        | Models                          | Meta-Model             | Checkpoint Dir            | Output Dir             |
| ------------- | ------------------------------- | ---------------------- | ------------------------- | ---------------------- |
| export_v13.py | 10 (LGB + CatBoost)             | Ridge                  | N/A (trains from scratch) | `mlops/artifacts/v13/` |
| export_v18.py | 36 (HGB + LGB + XGB + CatBoost) | HuberRegressor(ε=1.35) | `titan_checkpoints/`      | `mlops/artifacts/v18/` |
| export_v20.py | ~270 (same families, 10 seeds)  | HuberRegressor(ε=1.35) | `colossus_checkpoints/`   | `mlops/artifacts/v20/` |

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
docker compose up --build
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

### Verify the AI Mitigation Planner

Run the focused backend verification:

```bash
python -m pytest api/tests/test_counterfactual_planner.py -q
```

This checks the response contract, unknown-feature handling, `max_steps`, API registration, existing health/simulation imports, and that model artifacts remain unchanged.

Verify the frontend production build:

```bash
cd frontend
npm run build
```

After starting Docker Compose, inspect the endpoint through Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs).
