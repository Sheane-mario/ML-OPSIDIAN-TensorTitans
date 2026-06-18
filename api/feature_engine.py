"""
Feature Engineering Pipeline — Extracted from v13_solution.py

This module replicates the exact feature engineering pipeline used during
competition training. It transforms raw user inputs into the 115-feature
vector that the stacked ensemble expects.

The pipeline has 4 stages:
1. Date/Meta feature extraction
2. Domain-specific engineered features
3. Label encoding of categorical columns
4. Target encoding application (using pre-computed statistics)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import json
import os


# ─── Constants ──────────────────────────────────────────────────

CAT_COLS = [
    'district', 'landcover', 'soil_type', 'water_supply', 'electricity',
    'road_quality', 'urban_rural', 'water_presence_flag',
    'flood_occurrence_current_event', 'is_good_to_live',
    'reason_not_good_to_live', 'place_name'
]

DROP_COLS = ['record_id', 'gen_date', 'generation_date', 'is_synthetic', 'flood_risk_score']

TE_COLS_AND_SMOOTHING = [
    ('place_name', 5),
    ('district', 10),
    ('soil_type', 10),
    ('landcover', 10),
    ('road_quality', 10),
    ('flood_occurrence_current_event', 10),
]

TE_INTERACTION_DEFS = [
    ('rainfall_7d_mm',        'district', 'dist_te_x_rain'),
    ('extreme_weather_index', 'district', 'dist_te_x_extreme'),
    ('historical_flood_count','district', 'dist_te_x_flood'),
    ('rainfall_7d_mm',        'flood_occurrence_current_event', 'fl_te_x_rain'),
    ('extreme_weather_index', 'flood_occurrence_current_event', 'fl_te_x_extreme'),
    ('drainage_index',        'district', 'dist_te_x_drainage'),
    ('river_clip',            'district', 'dist_te_x_river'),
    ('elevation_m',           'district', 'dist_te_x_elev'),
    ('historical_flood_count','flood_occurrence_current_event', 'fl_te_x_flood'),
    ('drainage_index',        'flood_occurrence_current_event', 'fl_te_x_drainage'),
    ('rainfall_7d_mm',        'soil_type', 'soil_te_x_rain'),
    ('extreme_weather_index', 'soil_type', 'soil_te_x_extreme'),
]


class FeatureEngine:
    """
    Transforms raw input features into the model-ready feature vector.

    Usage:
        engine = FeatureEngine.from_artifacts("mlops/artifacts/v13/")
        features = engine.transform(raw_input_dict)
    """

    def __init__(
        self,
        label_encoders: Dict[str, Dict[str, int]],
        te_stats: Dict[str, Dict[str, float]],
        feature_columns: list,
        medians: Dict[str, float],
        global_mean: float
    ):
        self.label_encoders = label_encoders
        self.te_stats = te_stats
        self.feature_columns = feature_columns
        self.medians = medians
        self.global_mean = global_mean

    @classmethod
    def from_artifacts(cls, artifacts_dir: str) -> 'FeatureEngine':
        """Load a pre-fitted FeatureEngine from saved artifacts."""
        with open(os.path.join(artifacts_dir, "label_encoders.json"), "r") as f:
            label_encoders = json.load(f)
        with open(os.path.join(artifacts_dir, "te_stats.json"), "r") as f:
            te_stats = json.load(f)
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "r") as f:
            feature_columns = json.load(f)
        with open(os.path.join(artifacts_dir, "medians.json"), "r") as f:
            medians = json.load(f)
        with open(os.path.join(artifacts_dir, "global_mean.json"), "r") as f:
            global_mean = json.load(f)
        return cls(label_encoders, te_stats, feature_columns, medians, global_mean)

    def transform(self, raw_input: Dict[str, Any]) -> np.ndarray:
        """Transform a single raw input dict into a model-ready feature vector.

        Args:
            raw_input: Dictionary of raw feature values from the API request.

        Returns:
            1D numpy array of float32 values matching self.feature_columns.
        """
        # Convert to a single-row DataFrame
        df = pd.DataFrame([raw_input])

        # Stage 1: Date/Meta features
        df = self._add_date_features(df)

        # Stage 2: Domain-specific engineered features
        df = self._engineer_features(df)

        # Stage 3: Label encoding
        df = self._apply_label_encoding(df)

        # Stage 4: Target encoding + interactions
        df = self._apply_target_encoding(df)

        # Select and order features
        missing_cols = {col: self.medians.get(col, 0.0) for col in self.feature_columns if col not in df.columns}
        if missing_cols:
            # We assign to multiple columns at once instead of in a loop
            # to avoid Pandas PerformanceWarning: DataFrame is highly fragmented.
            missing_df = pd.DataFrame([missing_cols], index=df.index)
            df = pd.concat([df, missing_df], axis=1)

        df = df[self.feature_columns]

        # Fill NaNs with training medians
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(self.medians.get(col, 0.0))

        return df.values.astype(np.float32).flatten()

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from generation_date."""
        df = df.copy()
        df['gen_date'] = pd.to_datetime(df['generation_date'], errors='coerce')

        # If date parsing fails, use a sensible default
        if df['gen_date'].isna().any():
            df['gen_date'] = pd.Timestamp('2024-06-15')

        df['gen_month']       = df['gen_date'].dt.month
        df['gen_year']        = df['gen_date'].dt.year
        df['gen_day_of_year'] = df['gen_date'].dt.dayofyear
        df['gen_quarter']     = df['gen_date'].dt.quarter
        df['is_ne_monsoon']   = df['gen_month'].isin([12, 1, 2]).astype(int)
        df['is_sw_monsoon']   = df['gen_month'].isin([5, 6, 7, 8, 9]).astype(int)
        df['gen_month_sin']   = np.sin(2 * np.pi * df['gen_month'] / 12)
        df['gen_month_cos']   = np.cos(2 * np.pi * df['gen_month'] / 12)

        # Reason flags
        reason = df['reason_not_good_to_live'].fillna('Other')
        df['reason_flood_flag'] = reason.str.contains('flood', case=False).astype(int)
        df['reason_infra_flag'] = reason.str.contains('infrastructure', case=False).astype(int)
        df['reason_road_flag']  = reason.str.contains('road', case=False).astype(int)
        df['reason_other_flag'] = (reason == 'Other').astype(int)

        # Binary flags
        df['is_good_binary']    = (df['is_good_to_live'] == 'Yes').astype(int)
        df['log_inundation']    = np.log1p(df['inundation_area_sqm'])
        df['sqrt_inundation']   = np.sqrt(df['inundation_area_sqm'])
        df['inundation_per_pop'] = df['inundation_area_sqm'] / (df['population_density_per_km2'] + 1)

        # record_id_num — for simulation, use a dummy value
        df['record_id_num'] = 0

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replicate the engineer_features() function from v13_solution.py."""
        df = df.copy()
        eps = 1e-6

        df['rainfall_x_flood']        = df['rainfall_7d_mm'] * df['historical_flood_count']
        df['monthly_x_flood']         = df['monthly_rainfall_mm'] * df['historical_flood_count']
        df['rain_ratio']              = df['rainfall_7d_mm'] / (df['monthly_rainfall_mm'] + eps)
        df['rain_cum']                = df['rainfall_7d_mm'] + df['monthly_rainfall_mm']
        df['river_clip']              = df['distance_to_river_m'].clip(lower=0)
        df['river_rain_risk']         = df['rainfall_7d_mm'] / (df['river_clip'] + 1)
        df['river_monthly_risk']      = df['monthly_rainfall_mm'] / (df['river_clip'] + 1)
        df['elev_clip']               = df['elevation_m'].clip(lower=0)
        df['elev_rain_ratio']         = df['rainfall_7d_mm'] / (df['elev_clip'] + 1)
        df['low_elev_flag']           = (df['elevation_m'] < 30).astype(int)
        df['infra_socio']             = df['infrastructure_score'] * df['socioeconomic_status_index']
        df['water_veg_balance']       = df['ndwi'] - df['ndvi']
        df['ndvi_ndwi_product']       = df['ndvi'] * df['ndwi']
        df['ndwi_sq']                 = df['ndwi'] ** 2
        df['drainage_x_rain']         = df['drainage_index'] * df['rainfall_7d_mm']
        df['bad_drainage_rain']       = (df['drainage_index'] < 0.35).astype(int) * df['rainfall_7d_mm']
        df['urban_runoff']            = df['built_up_percent'] * df['rainfall_7d_mm'] / 100
        df['evac_hosp_sum']           = df['nearest_hospital_km'] + df['nearest_evac_km']
        df['max_dist_help']           = df[['nearest_hospital_km', 'nearest_evac_km']].max(axis=1)
        df['pop_x_rain']              = df['population_density_per_km2'] * df['rainfall_7d_mm']
        df['pop_x_flood']             = df['population_density_per_km2'] * df['historical_flood_count']
        df['extreme_x_rain']          = df['extreme_weather_index'] * df['rainfall_7d_mm']
        df['extreme_x_flood']         = df['extreme_weather_index'] * df['historical_flood_count']
        df['extreme_x_monthly']       = df['extreme_weather_index'] * df['monthly_rainfall_mm']
        df['seasonal_rain']           = df['seasonal_index'] * df['rainfall_7d_mm']
        df['seasonal_extreme']        = df['seasonal_index'] * df['extreme_weather_index']
        df['terrain_rain']            = df['terrain_roughness_index'] * df['rainfall_7d_mm']
        df['inundation_x_rain']       = df['inundation_area_sqm'] * df['rainfall_7d_mm']
        df['log_inundation_x_extreme'] = df['log_inundation'] * df['extreme_weather_index']
        df['composite_vuln']          = (
            df['rainfall_7d_mm'] * 0.3 +
            df['historical_flood_count'] * 15.0 +
            df['extreme_weather_index'] * 50.0 +
            (1 - df['drainage_index']) * 30.0 +
            df['built_up_percent'] * 0.10
        )

        return df

    def _apply_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-fitted label encoders to categorical columns."""
        df = df.copy()
        for col in CAT_COLS:
            if col in df.columns and col in self.label_encoders:
                le_map = self.label_encoders[col]
                df[col] = df[col].astype(str).fillna('missing').map(
                    lambda x, m=le_map: m.get(x, m.get('missing', 0))
                )
        return df

    def _apply_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-computed target encoding statistics.

        During training, fold-safe target encoding was used. For inference,
        we use the full-data TE statistics (computed on the entire training set).
        """
        df = df.copy()

        te_values = {}
        for col, _ in TE_COLS_AND_SMOOTHING:
            te_col = f'{col}_te'
            if col in self.te_stats:
                raw_val = str(int(df[col].iloc[0])) if col in df.columns else '0'
                te_val = self.te_stats[col].get(raw_val, self.global_mean)
                df[te_col] = te_val
                te_values[col] = te_val

        # TE interactions
        for feat_col, te_source, out_col in TE_INTERACTION_DEFS:
            te_val = te_values.get(te_source, self.global_mean)
            if feat_col in df.columns:
                df[out_col] = te_val * df[feat_col]
            else:
                df[out_col] = 0.0

        return df


def get_risk_level(score: float) -> str:
    """Convert a numeric flood risk score to a human-readable level."""
    if score < 0.25:
        return "Low"
    elif score < 0.50:
        return "Medium"
    elif score < 0.75:
        return "High"
    else:
        return "Critical"
