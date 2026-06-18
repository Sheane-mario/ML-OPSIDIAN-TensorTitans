"""
Pydantic schemas for strict input validation and response models.

All raw features from the training dataset are defined here with
appropriate bounds. The API will reject any request that violates
these constraints with a 422 Validation Error.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


# ─── Enums for Categorical Features ──────────────────────────────

class District(str, Enum):
    COLOMBO = "Colombo"
    GAMPAHA = "Gampaha"
    KALUTARA = "Kalutara"
    KANDY = "Kandy"
    MATALE = "Matale"
    NUWARA_ELIYA = "Nuwara Eliya"
    GALLE = "Galle"
    MATARA = "Matara"
    HAMBANTOTA = "Hambantota"
    JAFFNA = "Jaffna"
    KILINOCHCHI = "Kilinochchi"
    MANNAR = "Mannar"
    MULLAITIVU = "Mullaitivu"
    VAVUNIYA = "Vavuniya"
    BATTICALOA = "Batticaloa"
    AMPARA = "Ampara"
    TRINCOMALEE = "Trincomalee"
    KURUNEGALA = "Kurunegala"
    PUTTALAM = "Puttalam"
    ANURADHAPURA = "Anuradhapura"
    POLONNARUWA = "Polonnaruwa"
    BADULLA = "Badulla"
    MONARAGALA = "Monaragala"
    RATNAPURA = "Ratnapura"
    KEGALLE = "Kegalle"


class Landcover(str, Enum):
    URBAN = "Urban"
    AGRICULTURAL = "Agricultural"
    FOREST = "Forest"
    WATER_BODY = "Water Body"
    WETLAND = "Wetland"
    BARREN = "Barren"
    GRASSLAND = "Grassland"


class SoilType(str, Enum):
    CLAY = "Clay"
    SANDY = "Sandy"
    LOAM = "Loam"
    SILT = "Silt"
    LATERITE = "Laterite"
    ALLUVIAL = "Alluvial"
    RED_YELLOW_PODZOLIC = "Red-Yellow Podzolic"


class WaterSupply(str, Enum):
    PIPE_BORNE = "Pipe-borne"
    WELL = "Well"
    RIVER = "River"
    TANK = "Tank"
    BOREHOLE = "Borehole"
    SPRING = "Spring"


class RoadQuality(str, Enum):
    GOOD = "Good"
    MODERATE = "Moderate"
    POOR = "Poor"
    VERY_POOR = "Very Poor"


class UrbanRural(str, Enum):
    URBAN = "Urban"
    RURAL = "Rural"
    SEMI_URBAN = "Semi-Urban"


class FloodOccurrence(str, Enum):
    YES = "Yes"
    NO = "No"


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


class ModelVersion(str, Enum):
    V13 = "v13"
    V18_TITAN = "v18_titan"
    V20_COLOSSUS = "v20_colossus"


# ─── Simulation Request Schema ──────────────────────────────────

class SimulationRequest(BaseModel):
    """Input schema for the /api/simulate endpoint.

    Contains all raw features needed by the model. Features are
    grouped by category for clarity. Default values represent
    median/mode from the training data.
    """

    # --- Model Selection ---
    model_version: ModelVersion = Field(
        default=ModelVersion.V13,
        description="Which model version to use for prediction"
    )

    # --- Location / Categorical ---
    district: District = Field(
        default=District.COLOMBO,
        description="Administrative district in Sri Lanka"
    )
    landcover: Landcover = Field(
        default=Landcover.URBAN,
        description="Land cover type of the area"
    )
    soil_type: SoilType = Field(
        default=SoilType.CLAY,
        description="Predominant soil type"
    )
    water_supply: WaterSupply = Field(
        default=WaterSupply.PIPE_BORNE,
        description="Primary water supply source"
    )
    electricity: str = Field(
        default="Yes",
        description="Electricity availability"
    )
    road_quality: RoadQuality = Field(
        default=RoadQuality.MODERATE,
        description="Quality of roads in the area"
    )
    urban_rural: UrbanRural = Field(
        default=UrbanRural.URBAN,
        description="Urban/Rural classification"
    )
    water_presence_flag: str = Field(
        default="No",
        description="Presence of water body nearby"
    )
    flood_occurrence_current_event: FloodOccurrence = Field(
        default=FloodOccurrence.NO,
        description="Whether flooding is occurring in the current event"
    )
    is_good_to_live: YesNo = Field(
        default=YesNo.YES,
        description="Whether the area is considered good to live in"
    )
    reason_not_good_to_live: Optional[str] = Field(
        default="Other",
        description="Reason why area is not good to live (if applicable)"
    )
    place_name: str = Field(
        default="Unknown",
        description="Name of the place/location"
    )

    # --- Meteorological ---
    rainfall_7d_mm: float = Field(
        default=50.0, ge=0, le=1500,
        description="7-day cumulative rainfall in mm"
    )
    monthly_rainfall_mm: float = Field(
        default=200.0, ge=0, le=3000,
        description="Monthly cumulative rainfall in mm"
    )
    extreme_weather_index: float = Field(
        default=0.3, ge=0, le=1,
        description="Index of extreme weather intensity (0-1)"
    )
    seasonal_index: float = Field(
        default=0.5, ge=0, le=1,
        description="Seasonal weather index (0-1)"
    )

    # --- Geographical ---
    elevation_m: float = Field(
        default=30.0, ge=-10, le=2500,
        description="Elevation above sea level in meters"
    )
    distance_to_river_m: float = Field(
        default=500.0, ge=0, le=50000,
        description="Distance to nearest river in meters"
    )
    drainage_index: float = Field(
        default=0.5, ge=0, le=1,
        description="Drainage capacity index (0=poor, 1=excellent)"
    )
    terrain_roughness_index: float = Field(
        default=0.3, ge=0, le=1,
        description="Terrain roughness index (0=flat, 1=very rough)"
    )

    # --- Infrastructural ---
    infrastructure_score: float = Field(
        default=0.5, ge=0, le=1,
        description="Infrastructure quality score (0-1)"
    )
    nearest_hospital_km: float = Field(
        default=5.0, ge=0, le=100,
        description="Distance to nearest hospital in km"
    )
    nearest_evac_km: float = Field(
        default=3.0, ge=0, le=100,
        description="Distance to nearest evacuation center in km"
    )
    socioeconomic_status_index: float = Field(
        default=0.5, ge=0, le=1,
        description="Socioeconomic status index (0-1)"
    )
    population_density_per_km2: float = Field(
        default=500.0, ge=0, le=50000,
        description="Population density per square km"
    )

    # --- Satellite ---
    ndvi: float = Field(
        default=0.3, ge=-1, le=1,
        description="Normalized Difference Vegetation Index"
    )
    ndwi: float = Field(
        default=0.1, ge=-1, le=1,
        description="Normalized Difference Water Index"
    )

    # --- Other ---
    built_up_percent: float = Field(
        default=30.0, ge=0, le=100,
        description="Percentage of built-up area"
    )
    historical_flood_count: int = Field(
        default=2, ge=0, le=100,
        description="Number of historical flood events recorded"
    )
    inundation_area_sqm: float = Field(
        default=1000.0, ge=0,
        description="Inundation area in square meters"
    )
    generation_date: str = Field(
        default="2024-06-15",
        description="Date of data generation (YYYY-MM-DD)"
    )
    is_synthetic: int = Field(
        default=0, ge=0, le=1,
        description="Whether the data point is synthetic"
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "district": "Colombo",
            "rainfall_7d_mm": 120.0,
            "drainage_index": 0.3,
            "infrastructure_score": 0.6,
            "nearest_hospital_km": 3.0,
            "model_version": "v13"
        }]
    }}


# ─── Simulation Response Schema ─────────────────────────────────

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class SimulationResponse(BaseModel):
    """Output schema for the /api/simulate endpoint."""
    flood_risk_score: float = Field(
        description="Predicted flood risk score (0.0 - 1.0)"
    )
    risk_level: str = Field(
        description="Human-readable risk level (Low/Medium/High/Critical)"
    )
    model_version: str = Field(
        description="Model version used for this prediction"
    )
    inference_time_ms: float = Field(
        description="Time taken for inference in milliseconds"
    )
    feature_importance: List[FeatureImportanceItem] = Field(
        default=[],
        description="Top features driving this prediction"
    )


# ─── Other Response Schemas ──────────────────────────────────────

class ModelInfo(BaseModel):
    version: str
    name: str
    description: str
    is_default: bool
    num_base_models: int


class DistrictDefaults(BaseModel):
    district: str
    defaults: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    database_connected: bool


class AnalyticsResponse(BaseModel):
    total_simulations: int
    avg_latency_ms: float
    avg_risk_score: float
    most_tweaked_features: Dict[str, int]
    score_distribution: Dict[str, int]
    recent_simulations: List[Dict]


# ─── Urban Planning Schemas ─────────────────────────────────────

class InterventionItem(BaseModel):
    """A single actionable intervention with its impact."""
    feature: str = Field(description="Feature name")
    feature_label: str = Field(description="Human-readable label")
    current_value: float = Field(description="Current feature value")
    improved_value: float = Field(description="Recommended improved value")
    baseline_score: float = Field(description="Score before intervention")
    improved_score: float = Field(description="Score after intervention")
    risk_reduction_pct: float = Field(description="Percentage reduction in risk")
    category: str = Field(description="Intervention category")


class SensitivityItem(BaseModel):
    """Sensitivity analysis for a single feature."""
    feature: str
    label: str
    values: List[float] = Field(description="Array of test values")
    scores: List[float] = Field(description="Corresponding risk scores")


class InterventionPlannerResponse(BaseModel):
    """Response for the intervention planner endpoint."""
    district: str
    model_version: str
    baseline_score: float
    baseline_risk_level: str
    interventions: List[InterventionItem]
    sensitivity: List[SensitivityItem]


class ScenarioInput(BaseModel):
    """Input for a single scenario in comparison."""
    label: str = Field(default="Scenario", description="Scenario label")
    district: str = Field(default="Colombo")
    model_version: str = Field(default="v13")
    # Numerical features - all optional
    rainfall_7d_mm: Optional[float] = None
    monthly_rainfall_mm: Optional[float] = None
    drainage_index: Optional[float] = None
    infrastructure_score: Optional[float] = None
    elevation_m: Optional[float] = None
    distance_to_river_m: Optional[float] = None
    nearest_hospital_km: Optional[float] = None
    nearest_evac_km: Optional[float] = None
    built_up_percent: Optional[float] = None
    extreme_weather_index: Optional[float] = None
    seasonal_index: Optional[float] = None
    terrain_roughness_index: Optional[float] = None
    historical_flood_count: Optional[int] = None
    population_density_per_km2: Optional[float] = None
    ndvi: Optional[float] = None
    ndwi: Optional[float] = None
    socioeconomic_status_index: Optional[float] = None
    inundation_area_sqm: Optional[float] = None


class ScenarioResult(BaseModel):
    """Result for a single scenario."""
    label: str
    district: str
    flood_risk_score: float
    risk_level: str
    features_used: Dict[str, Any]


class ScenarioComparisonRequest(BaseModel):
    """Request body for scenario comparison."""
    scenario_a: ScenarioInput
    scenario_b: ScenarioInput


class FeatureDelta(BaseModel):
    """Delta for a single feature between two scenarios."""
    feature: str
    label: str
    value_a: float
    value_b: float
    delta: float
    impact_direction: str  # "positive" or "negative" or "neutral"


class ScenarioComparisonResponse(BaseModel):
    """Response for scenario comparison."""
    scenario_a: ScenarioResult
    scenario_b: ScenarioResult
    score_delta: float
    risk_improvement: str  # "improved", "worsened", "unchanged"
    feature_deltas: List[FeatureDelta]


class DistrictRiskItem(BaseModel):
    """Risk overview for a single district."""
    district: str
    flood_risk_score: float
    risk_level: str
    key_factors: Dict[str, float]  # Top contributing features


class DistrictOverviewResponse(BaseModel):
    """Response for the district overview."""
    model_version: str
    districts: List[DistrictRiskItem]
    highest_risk: str
    lowest_risk: str
    avg_risk_score: float


# ─── Counterfactual Planner Schemas ──────────────────────────────────────────

class OptimizePlanRequest(BaseModel):
    """Constraints for generating a counterfactual mitigation plan."""
    district: District = Field(default=District.COLOMBO)
    model_version: ModelVersion = Field(default=ModelVersion.V13)
    target_risk_level: Literal["Low", "Medium", "High", "Critical"] = Field(
        default="Medium"
    )
    max_steps: int = Field(default=3, ge=1, le=5)
    budget_profile: Literal["low_cost", "balanced", "aggressive"] = Field(
        default="balanced"
    )
    allowed_features: Optional[List[str]] = Field(
        default=None,
        description=(
            "Actionable features available to the optimizer. "
            "When omitted, all supported actionable features are used."
        ),
    )


class PlanStep(BaseModel):
    """A single accepted feature change in an optimized mitigation plan."""
    feature: str
    feature_label: str
    category: str
    from_value: float
    to_value: float
    score_before: float
    score_after: float
    absolute_reduction: float
    relative_reduction_pct: float
    cost_level: str
    rationale: str


class SearchTraceItem(BaseModel):
    """One candidate evaluation recorded during optimization."""
    step: int
    tried_feature: str
    tried_value: float
    resulting_score: float
    resulting_risk_level: Literal["Low", "Medium", "High", "Critical"]
    accepted: bool


class AlternativePlan(BaseModel):
    """A ranked alternative to the recommended mitigation plan."""
    name: str
    optimized_score: float
    optimized_risk_level: Literal["Low", "Medium", "High", "Critical"]
    target_reached: bool
    risk_reduction_pct: float
    steps: List[PlanStep]


class OptimizePlanResponse(BaseModel):
    """Result of the counterfactual mitigation plan search."""
    district: str
    model_version: str
    baseline_score: float
    baseline_risk_level: Literal["Low", "Medium", "High", "Critical"]
    target_risk_level: Literal["Low", "Medium", "High", "Critical"]
    target_score_threshold: float
    target_reached: bool
    optimized_score: float
    optimized_risk_level: Literal["Low", "Medium", "High", "Critical"]
    risk_reduction_pct: float
    recommended_plan: List[PlanStep]
    search_trace: List[SearchTraceItem]
    alternatives: List[AlternativePlan]
