"""Black-box counterfactual search for flood mitigation plans.

The planner treats the existing prediction pipeline as an immutable scorer.
It starts from district defaults, evaluates bounded actionable changes, and
greedily selects the best budget-adjusted improvement at each step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from api.feature_engine import get_risk_level
from api.model_manager import model_manager


TARGET_THRESHOLDS = {
    "Low": 0.25,
    "Medium": 0.50,
    "High": 0.75,
    "Critical": 1.00,
}

SUPPORTED_BUDGET_PROFILES = {"low_cost", "balanced", "aggressive"}


@dataclass(frozen=True)
class ActionConfig:
    label: str
    category: str
    target_value: float
    higher_is_better: bool
    cost_level: str
    cost_weight: float
    rationale: str


ACTIONABLE_FEATURES: dict[str, ActionConfig] = {
    "drainage_index": ActionConfig(
        label="Drainage System",
        category="Infrastructure",
        target_value=0.85,
        higher_is_better=True,
        cost_level="low",
        cost_weight=1.0,
        rationale=(
            "Improve drainage capacity to move stormwater away from exposed areas."
        ),
    ),
    "infrastructure_score": ActionConfig(
        label="Infrastructure Score",
        category="Infrastructure",
        target_value=0.90,
        higher_is_better=True,
        cost_level="low",
        cost_weight=1.1,
        rationale=(
            "Strengthen protective infrastructure and improve service resilience."
        ),
    ),
    "nearest_evac_km": ActionConfig(
        label="Evacuation Center Distance",
        category="Emergency",
        target_value=1.0,
        higher_is_better=False,
        cost_level="medium",
        cost_weight=2.0,
        rationale=(
            "Reduce access distance by adding or relocating evacuation capacity."
        ),
    ),
    "nearest_hospital_km": ActionConfig(
        label="Hospital Distance",
        category="Healthcare",
        target_value=1.5,
        higher_is_better=False,
        cost_level="high",
        cost_weight=3.0,
        rationale=(
            "Improve emergency healthcare access for flood-affected communities."
        ),
    ),
    "distance_to_river_m": ActionConfig(
        label="River Setback",
        category="Geography",
        target_value=1500.0,
        higher_is_better=True,
        cost_level="high",
        cost_weight=3.2,
        rationale=(
            "Increase effective river setback through zoning or protective relocation."
        ),
    ),
    "built_up_percent": ActionConfig(
        label="Built-Up Area",
        category="Land Use",
        target_value=10.0,
        higher_is_better=False,
        cost_level="low",
        cost_weight=1.2,
        rationale=(
            "Reduce impermeable coverage and create more space for water absorption."
        ),
    ),
}


PROFILE_FRACTIONS = {
    "low_cost": (0.25, 0.50),
    "balanced": (0.33, 0.66, 1.00),
    "aggressive": (0.50, 0.75, 1.00),
}

LOW_COST_PREFERRED = {
    "drainage_index",
    "infrastructure_score",
    "built_up_percent",
}

EPSILON = 1e-9


def _build_baseline_features(district: str) -> dict[str, Any]:
    """Reuse the API's canonical district-default feature builder lazily.

    The lazy import keeps this module reusable and prevents a module-level
    cycle when ``api.main`` later imports the planner to expose an endpoint.
    Unknown districts are already handled by ``_build_raw_features`` through
    its generic-default fallback.
    """
    from api.main import _build_raw_features

    return _build_raw_features(district, {})


def _score(features: dict[str, Any], model_version: str) -> float:
    score, _, _ = model_manager.predict(dict(features), model_version)
    return float(score)


def _target_reached(score: float, threshold: float) -> bool:
    return score < threshold


def _candidate_values(
    current_value: float,
    config: ActionConfig,
    budget_profile: str,
) -> list[float]:
    """Generate deterministic values between the current and desired state."""
    target = config.target_value
    already_at_or_beyond_target = (
        config.higher_is_better and current_value >= target
    ) or (
        not config.higher_is_better and current_value <= target
    )
    if already_at_or_beyond_target or math.isclose(
        current_value, target, rel_tol=0.0, abs_tol=EPSILON
    ):
        return []

    fractions = PROFILE_FRACTIONS[budget_profile]
    values = {
        round(current_value + (target - current_value) * fraction, 6)
        for fraction in fractions
    }

    # Never propose movement beyond the configured target.
    lower, upper = sorted((current_value, target))
    return sorted(
        (value for value in values if lower <= value <= upper),
        reverse=target < current_value,
    )


def _budget_utility(
    reduction: float,
    feature: str,
    config: ActionConfig,
    budget_profile: str,
) -> float:
    """Adjust raw risk reduction according to the selected cost profile."""
    if budget_profile == "aggressive":
        return reduction

    if budget_profile == "low_cost":
        preference = 1.35 if feature in LOW_COST_PREFERRED else 0.75
        return reduction * preference / config.cost_weight

    return reduction / math.sqrt(config.cost_weight)


def _relative_reduction(score_before: float, score_after: float) -> float:
    if score_before <= 0:
        return 0.0
    return ((score_before - score_after) / score_before) * 100.0


def _make_plan_step(
    feature: str,
    config: ActionConfig,
    from_value: float,
    to_value: float,
    score_before: float,
    score_after: float,
) -> dict[str, Any]:
    return {
        "feature": feature,
        "feature_label": config.label,
        "category": config.category,
        "from_value": round(float(from_value), 4),
        "to_value": round(float(to_value), 4),
        "score_before": round(score_before, 4),
        "score_after": round(score_after, 4),
        "absolute_reduction": round(score_before - score_after, 4),
        "relative_reduction_pct": round(
            _relative_reduction(score_before, score_after), 2
        ),
        "cost_level": config.cost_level,
        "rationale": config.rationale,
    }


def _make_alternative(
    name: str,
    baseline_score: float,
    optimized_score: float,
    threshold: float,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": name,
        "optimized_score": round(optimized_score, 4),
        "optimized_risk_level": get_risk_level(optimized_score),
        "target_reached": _target_reached(optimized_score, threshold),
        "risk_reduction_pct": round(
            _relative_reduction(baseline_score, optimized_score), 2
        ),
        "steps": steps,
    }


def generate_counterfactual_plan(
    district: str,
    model_version: str,
    target_risk_level: str,
    max_steps: int = 3,
    budget_profile: str = "balanced",
    allowed_features: list[str] | None = None,
) -> dict:
    """Generate a bounded counterfactual mitigation plan.

    Unknown allowed features are ignored. Unknown districts are passed to the
    existing baseline builder, which applies generic defaults. The search is
    greedy and deterministic: every iteration evaluates all candidate values
    for each unused supported feature and accepts the best budget-adjusted
    score reduction.
    """
    if target_risk_level not in TARGET_THRESHOLDS:
        raise ValueError(
            f"Unsupported target_risk_level: {target_risk_level!r}"
        )
    if budget_profile not in SUPPORTED_BUDGET_PROFILES:
        raise ValueError(f"Unsupported budget_profile: {budget_profile!r}")

    max_steps = max(1, min(int(max_steps), 5))
    target_threshold = TARGET_THRESHOLDS[target_risk_level]

    if allowed_features is None:
        selected_features = list(ACTIONABLE_FEATURES)
    else:
        requested = set(allowed_features)
        selected_features = [
            feature for feature in ACTIONABLE_FEATURES if feature in requested
        ]

    baseline_features = _build_baseline_features(district)
    baseline_score = _score(baseline_features, model_version)
    baseline_level = get_risk_level(baseline_score)

    current_features = dict(baseline_features)
    current_score = baseline_score
    recommended_plan: list[dict[str, Any]] = []
    search_trace: list[dict[str, Any]] = []
    alternative_candidates: list[dict[str, Any]] = []
    used_features: set[str] = set()
    search_stalled = False

    if not _target_reached(current_score, target_threshold):
        for step_number in range(1, max_steps + 1):
            candidates: list[dict[str, Any]] = []

            for feature in selected_features:
                if feature in used_features:
                    continue

                config = ACTIONABLE_FEATURES[feature]
                from_value = float(current_features[feature])
                for tried_value in _candidate_values(
                    from_value, config, budget_profile
                ):
                    candidate_features = dict(current_features)
                    candidate_features[feature] = tried_value
                    resulting_score = _score(candidate_features, model_version)
                    reduction = current_score - resulting_score

                    candidates.append(
                        {
                            "feature": feature,
                            "config": config,
                            "from_value": from_value,
                            "to_value": tried_value,
                            "score": resulting_score,
                            "reduction": reduction,
                            "utility": _budget_utility(
                                reduction,
                                feature,
                                config,
                                budget_profile,
                            ),
                        }
                    )

            improving = [
                candidate
                for candidate in candidates
                if candidate["reduction"] > EPSILON
            ]
            improving.sort(
                key=lambda candidate: (
                    -candidate["utility"],
                    -candidate["reduction"],
                    candidate["config"].cost_weight,
                    candidate["feature"],
                    candidate["to_value"],
                )
            )
            accepted = improving[0] if improving else None

            for candidate in candidates:
                is_accepted = candidate is accepted
                search_trace.append(
                    {
                        "step": step_number,
                        "tried_feature": candidate["feature"],
                        "tried_value": round(candidate["to_value"], 4),
                        "resulting_score": round(candidate["score"], 4),
                        "resulting_risk_level": get_risk_level(
                            candidate["score"]
                        ),
                        "accepted": is_accepted,
                    }
                )

                if candidate["reduction"] > EPSILON and not is_accepted:
                    alternative_step = _make_plan_step(
                        candidate["feature"],
                        candidate["config"],
                        candidate["from_value"],
                        candidate["to_value"],
                        current_score,
                        candidate["score"],
                    )
                    alternative_candidates.append(
                        _make_alternative(
                            name=(
                                f"Alternative via "
                                f"{candidate['config'].label}"
                            ),
                            baseline_score=baseline_score,
                            optimized_score=candidate["score"],
                            threshold=target_threshold,
                            steps=[*recommended_plan, alternative_step],
                        )
                    )

            if accepted is None:
                search_stalled = True
                break

            accepted_step = _make_plan_step(
                accepted["feature"],
                accepted["config"],
                accepted["from_value"],
                accepted["to_value"],
                current_score,
                accepted["score"],
            )
            recommended_plan.append(accepted_step)
            current_features[accepted["feature"]] = accepted["to_value"]
            current_score = accepted["score"]
            used_features.add(accepted["feature"])

            if _target_reached(current_score, target_threshold):
                break

    target_reached = _target_reached(current_score, target_threshold)

    # Keep distinct, best-scoring alternatives. The name communicates the
    # graceful no-improvement outcome while remaining schema-compatible.
    unique_alternatives: dict[tuple[str, ...], dict[str, Any]] = {}
    for alternative in alternative_candidates:
        signature = tuple(step["feature"] for step in alternative["steps"])
        existing = unique_alternatives.get(signature)
        if (
            existing is None
            or alternative["optimized_score"] < existing["optimized_score"]
        ):
            unique_alternatives[signature] = alternative

    alternatives = sorted(
        unique_alternatives.values(),
        key=lambda alternative: (
            not alternative["target_reached"],
            alternative["optimized_score"],
            len(alternative["steps"]),
            alternative["name"],
        ),
    )[:3]

    if not target_reached and search_stalled:
        stalled_explanation = _make_alternative(
            name=(
                "No further supported candidate reduced the predicted risk; "
                "review constraints or allowed features."
            ),
            baseline_score=baseline_score,
            optimized_score=current_score,
            threshold=target_threshold,
            steps=list(recommended_plan),
        )
        alternatives = [stalled_explanation, *alternatives][:3]

    return {
        "district": district,
        "model_version": model_version,
        "baseline_score": round(baseline_score, 4),
        "baseline_risk_level": baseline_level,
        "target_risk_level": target_risk_level,
        "target_score_threshold": target_threshold,
        "target_reached": target_reached,
        "optimized_score": round(current_score, 4),
        "optimized_risk_level": get_risk_level(current_score),
        "risk_reduction_pct": round(
            _relative_reduction(baseline_score, current_score), 2
        ),
        "recommended_plan": recommended_plan,
        "search_trace": search_trace,
        "alternatives": alternatives,
    }
