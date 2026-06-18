# AI Counterfactual Flood Mitigation Planner

## Purpose

The AI Mitigation Planner is a decision-support simulation tool. It searches for a small set of controllable feature changes that reduce the existing model's predicted flood-risk score for a district.

It does not guarantee flood prevention, establish causal effects, or replace engineering and policy review. It does not retrain, modify, or replace the prediction model, feature pipeline, MLflow setup, or model artifacts.

## User Flow

From `/optimize`, a user selects:

- one of the 25 supported districts;
- a target risk level: `Low`, `Medium`, or `High`;
- a budget profile: `low_cost`, `balanced`, or `aggressive`;
- a maximum of 1 to 5 plan steps; and
- the actionable features the planner may change.

The page shows baseline and optimized risk, target status, recommended steps, the candidate search trace, and alternative plans.

## API Contract

### Endpoint

`POST /api/optimize-plan`

### Request

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

`target_risk_level` accepts `Low`, `Medium`, `High`, or `Critical`. The frontend intentionally offers the first three because a `Critical` target uses a threshold of `1.0` and is rarely useful. `allowed_features` is optional; omitting it enables all supported actionable features. Unknown feature names are ignored safely.

### Response

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

Scores in this example are illustrative. If the target is not reached, the response returns the best improving plan found within `max_steps` with `target_reached: false`.

## Risk Thresholds

| Target | Required score |
|---|---:|
| Low | `< 0.25` |
| Medium | `< 0.50` |
| High | `< 0.75` |
| Critical | `< 1.00` |

## Search Algorithm

The implementation uses deterministic, bounded greedy search:

1. Build a complete baseline from district defaults. Unknown districts use generic defaults.
2. Score the baseline through the unchanged `model_manager.predict(...)` path.
3. Generate bounded candidate values for each allowed, unused actionable feature.
4. Score every candidate as a black-box counterfactual.
5. Select the best improving candidate after applying the budget-profile preference.
6. Repeat until the target is reached, `max_steps` is exhausted, or no candidate improves risk.

The supported actions are:

- `drainage_index`
- `infrastructure_score`
- `nearest_evac_km`
- `nearest_hospital_km`
- `distance_to_river_m`
- `built_up_percent`

The `low_cost` profile favors drainage, infrastructure, and built-up-area actions. `balanced` trades off reduction against cost weight. `aggressive` prioritizes maximum raw score reduction.

## Implementation Files

- `api/schemas.py` — request and response validation.
- `api/counterfactual_planner.py` — candidate generation and black-box search.
- `api/main.py` — `POST /api/optimize-plan`.
- `api/tests/test_counterfactual_planner.py` — focused verification and artifact-integrity checks.
- `frontend/src/app/optimize/page.tsx` — planner UI.
- `frontend/src/components/Navbar.tsx` — `/optimize` navigation.

## Running and Verification

Start the full stack:

```bash
docker compose up --build
```

Then open:

- Planner UI: [http://localhost:3000/optimize](http://localhost:3000/optimize)
- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

Run focused backend verification:

```bash
python -m pytest api/tests/test_counterfactual_planner.py -q
```

Run the frontend production build:

```bash
cd frontend
npm run build
```

The backend verification checks response fields, unknown-feature handling, step limits, API registration, existing health/simulation imports, and SHA-256 equality of model artifacts before and after planner execution.
