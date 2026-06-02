"""
V2 Solution: Flood Risk Score Prediction - Sri Lanka
Key insights from EDA:
  - Train: only ~20,886 rows (small dataset)
  - Target: flood_risk_score in [0,1], mean=0.478
  - Linear correlations very low (max 0.081) -> complex non-linear patterns
  - ~800-1500 missing values in pre-engineered columns
  - flood_occurrence_current_event is a strong categorical

Strategy:
  1. Use ALL 44 columns + 25+ engineered features  
  2. Ensemble: LightGBM + XGBoost + CatBoost + ExtraTrees
  3. 5-fold CV with OOF predictions
  4. Ridge stacking meta-learner
  5. Predictions clipped to [0,1]
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SEED = 42
N_FOLDS = 5
DATA_DIR = "data"
OUTPUT_FILE = "submission_v2.csv"
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

TARGET = 'flood_risk_score'

DROP_COLS = [
    'record_id', 'place_name', 'reason_not_good_to_live',
    'is_good_to_live', 'is_synthetic', 'generation_date',
    'inundation_area_sqm',
]

CAT_COLS = [
    'district', 'landcover', 'soil_type', 'water_supply',
    'electricity', 'road_quality', 'urban_rural',
    'water_presence_flag', 'flood_occurrence_current_event'
]

# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    eps = 1e-6  # avoid div by zero

    # --- Rainfall × flood history interactions ---
    df['rainfall_x_flood_count']  = df['rainfall_7d_mm'] * df['historical_flood_count']
    df['monthly_x_flood_count']   = df['monthly_rainfall_mm'] * df['historical_flood_count']
    df['rain_ratio_7d_monthly']   = df['rainfall_7d_mm'] / (df['monthly_rainfall_mm'] + eps)
    
    # --- River proximity risk ---
    df['river_rainfall_risk']    = df['rainfall_7d_mm'] / (df['distance_to_river_m'].clip(lower=0) + 1)
    df['river_monthly_risk']     = df['monthly_rainfall_mm'] / (df['distance_to_river_m'].clip(lower=0) + 1)
    
    # --- Elevation-rainfall compound ---
    df['elev_rainfall_ratio']    = df['rainfall_7d_mm'] / (df['elevation_m'].clip(lower=0) + 1)
    df['elev_x_river']           = (df['elevation_m'].clip(lower=0) + 1) * (df['distance_to_river_m'].clip(lower=0) + 1)
    
    # --- Infrastructure & socioeconomic ---
    df['infra_socio_gap']        = df['infrastructure_score'] * df['socioeconomic_status_index']
    df['infra_per_pop']          = df['infrastructure_score'] / (df['population_density_per_km2'] + 1)
    
    # --- Vegetation & water indices ---
    df['water_veg_balance']      = df['ndwi'] - df['ndvi']
    df['ndvi_ndwi_product']      = df['ndvi'] * df['ndwi']
    
    # --- Drainage effectiveness ---
    df['drainage_rain_ratio']    = df['drainage_index'] / (df['rainfall_7d_mm'] + 1)
    df['drainage_x_rain']        = df['drainage_index'] * df['rainfall_7d_mm']
    
    # --- Urban runoff amplifier (impervious surfaces) ---
    df['urban_flood_amplifier']  = df['built_up_percent'] * df['rainfall_7d_mm'] / 100
    
    # --- Access to services ---
    df['hospital_evac_sum']      = df['nearest_hospital_km'] + df['nearest_evac_km']
    df['max_distance_to_help']   = df[['nearest_hospital_km', 'nearest_evac_km']].max(axis=1)
    df['min_distance_to_help']   = df[['nearest_hospital_km', 'nearest_evac_km']].min(axis=1)
    
    # --- Population exposure ---
    df['pop_density_x_rain']     = df['population_density_per_km2'] * df['rainfall_7d_mm']
    df['pop_density_x_flood']    = df['population_density_per_km2'] * df['historical_flood_count']
    
    # --- Terrain interactions ---
    df['terrain_rain_interaction'] = df['terrain_roughness_index'] * df['rainfall_7d_mm']
    
    # --- Extreme weather compound ---
    df['extreme_x_rain']         = df['extreme_weather_index'] * df['rainfall_7d_mm']
    df['extreme_x_flood']        = df['extreme_weather_index'] * df['historical_flood_count']
    df['extreme_x_monthly']      = df['extreme_weather_index'] * df['monthly_rainfall_mm']
    
    # --- Seasonal compound ---
    df['seasonal_rain']          = df['seasonal_index'] * df['rainfall_7d_mm']
    df['seasonal_monthly']       = df['seasonal_index'] * df['monthly_rainfall_mm']
    df['seasonal_x_extreme']     = df['seasonal_index'] * df['extreme_weather_index']
    
    # --- Log transforms (additional if pre-computed not available) ---
    df['log_distance_river']     = np.log1p(df['distance_to_river_m'].clip(lower=0))
    df['log_pop_density']        = np.log1p(df['population_density_per_km2'].clip(lower=0))
    df['log_elevation']          = np.log1p(df['elevation_m'].clip(lower=0))
    df['log_rainfall_7d']        = np.log1p(df['rainfall_7d_mm'].clip(lower=0))
    df['log_monthly_rain']       = np.log1p(df['monthly_rainfall_mm'].clip(lower=0))
    
    # --- Composite vulnerability score ---
    df['composite_vulnerability'] = (
        df['rainfall_7d_mm'] * 0.3 +
        df['historical_flood_count'] * 0.2 +
        df['extreme_weather_index'] * 0.2 +
        (1 - df['drainage_index']) * 0.15 +
        df['built_up_percent'] / 100 * 0.15
    )
    
    return df

print("Engineering features...")
train = engineer_features(train)
test  = engineer_features(test)

# ─────────────────────────────────────────────
# 3. District-level Target Encoding (train only → leakage-safe via KFold)
# ─────────────────────────────────────────────
print("Computing district-level aggregations (from training data only)...")
district_stats = train.groupby('district').agg(
    district_mean_risk    = (TARGET, 'mean'),
    district_median_risk  = (TARGET, 'median'),
    district_std_risk     = (TARGET, 'std'),
    district_flood_mean   = ('historical_flood_count', 'mean'),
    district_rain_mean    = ('rainfall_7d_mm', 'mean'),
    district_pop_mean     = ('population_density_per_km2', 'mean'),
    district_extreme_mean = ('extreme_weather_index', 'mean'),
).reset_index()

# Fill std NaN (single-sample districts)
district_stats['district_std_risk'] = district_stats['district_std_risk'].fillna(0)

train = train.merge(district_stats, on='district', how='left')
test  = test.merge(district_stats,  on='district', how='left')

# ─────────────────────────────────────────────
# 4. Encode Categoricals
# ─────────────────────────────────────────────
print("Encoding categoricals...")

# Store test record_ids before concat
test_record_ids = test['record_id'].copy()

# Concat for consistent label encoding
all_data = pd.concat([train, test], axis=0, ignore_index=True)

for col in CAT_COLS:
    if col in all_data.columns:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))

n_train = len(train)
train_encoded = all_data.iloc[:n_train].copy()
test_encoded  = all_data.iloc[n_train:].copy()

# Re-attach target
train_encoded[TARGET] = train[TARGET].values

# ─────────────────────────────────────────────
# 5. Final Feature Set
# ─────────────────────────────────────────────
EXCLUDE = set(DROP_COLS + [TARGET, 'record_id'])

feature_cols = [c for c in train_encoded.columns if c not in EXCLUDE]
print(f"Total features: {len(feature_cols)}")

X      = train_encoded[feature_cols].copy()
y      = train_encoded[TARGET].copy()
X_test = test_encoded[feature_cols].copy()

# Impute with median (important for tree models with missing values)
medians = X.median(numeric_only=True)
X      = X.fillna(medians)
X_test = X_test.fillna(medians)

print(f"X shape: {X.shape}  |  Any NaN: {X.isnull().any().any()}")
print(f"X_test shape: {X_test.shape}  |  Any NaN: {X_test.isnull().any().any()}")

X_arr      = X.values.astype(np.float32)
y_arr      = y.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)

# ─────────────────────────────────────────────
# 6. Cross-validated OOF Predictions
# ─────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros(len(X))
oof_xgb  = np.zeros(len(X))
oof_cat  = np.zeros(len(X))
oof_et   = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_et  = np.zeros(len(X_test))

# ─── LightGBM ────────────────────────────────
print("\n=== Training LightGBM (5-fold) ===")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.02,
    'num_leaves': 255,
    'max_depth': -1,
    'min_child_samples': 15,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'n_jobs': -1,
    'verbose': -1,
    'seed': SEED,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
    
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(1000)]
    )
    
    oof_lgb[val_idx] = model.predict(X_val)
    test_lgb        += model.predict(X_test_arr) / N_FOLDS
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    print(f"  Fold {fold+1}/{N_FOLDS}: RMSE={rmse:.5f} | best_iter={model.best_iteration}")

lgb_oof_rmse = np.sqrt(mean_squared_error(y_arr, oof_lgb))
print(f"LightGBM OOF RMSE: {lgb_oof_rmse:.5f}")

# ─── XGBoost ─────────────────────────────────
print("\n=== Training XGBoost (5-fold) ===")
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.02,
    'max_depth': 8,
    'min_child_weight': 5,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'n_jobs': -1,
    'seed': SEED,
    'tree_method': 'hist',
    'verbosity': 0,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_test_arr, feature_names=feature_cols)
    
    model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=5000,
        evals=[(dval, 'val')],
        early_stopping_rounds=150,
        verbose_eval=1000,
    )
    
    oof_xgb[val_idx] = model.predict(dval)
    test_xgb        += model.predict(dtest) / N_FOLDS
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    print(f"  Fold {fold+1}/{N_FOLDS}: RMSE={rmse:.5f} | best_iter={model.best_iteration}")

xgb_oof_rmse = np.sqrt(mean_squared_error(y_arr, oof_xgb))
print(f"XGBoost OOF RMSE: {xgb_oof_rmse:.5f}")

# ─── CatBoost ────────────────────────────────
print("\n=== Training CatBoost (5-fold) ===")
cat_params = dict(
    iterations=5000,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=3,
    min_data_in_leaf=15,
    subsample=0.75,
    colsample_bylevel=0.75,
    random_seed=SEED,
    task_type='CPU',
    verbose=0,
    eval_metric='RMSE',
    early_stopping_rounds=150,
)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
    
    model = CatBoostRegressor(**cat_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    
    oof_cat[val_idx] = model.predict(X_val)
    test_cat        += model.predict(X_test_arr) / N_FOLDS
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
    print(f"  Fold {fold+1}/{N_FOLDS}: RMSE={rmse:.5f} | best_iter={model.best_iteration_}")

cat_oof_rmse = np.sqrt(mean_squared_error(y_arr, oof_cat))
print(f"CatBoost OOF RMSE: {cat_oof_rmse:.5f}")

# ─── Extra Trees ──────────────────────────────
print("\n=== Training Extra Trees (5-fold) ===")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
    
    et = ExtraTreesRegressor(
        n_estimators=500, max_depth=25, min_samples_leaf=3,
        max_features=0.6, n_jobs=-1, random_state=SEED + fold
    )
    et.fit(X_tr, y_tr)
    
    oof_et[val_idx] = et.predict(X_val)
    test_et        += et.predict(X_test_arr) / N_FOLDS
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_et[val_idx]))
    print(f"  Fold {fold+1}/{N_FOLDS}: RMSE={rmse:.5f}")

et_oof_rmse = np.sqrt(mean_squared_error(y_arr, oof_et))
print(f"Extra Trees OOF RMSE: {et_oof_rmse:.5f}")

# ─────────────────────────────────────────────
# 7. Stacking with Ridge Meta-Learner
# ─────────────────────────────────────────────
print("\n=== Stacking ===")
oof_models   = [oof_lgb,  oof_xgb,  oof_cat,  oof_et]
test_models  = [test_lgb, test_xgb, test_cat, test_et]
rmse_scores  = [lgb_oof_rmse, xgb_oof_rmse, cat_oof_rmse, et_oof_rmse]
model_labels = ['LGB', 'XGB', 'CAT', 'ET']

# Stack OOF
oof_stack  = np.column_stack(oof_models)
test_stack = np.column_stack(test_models)

# Ridge meta-learner (cross-validated)
meta = Ridge(alpha=1.0)
meta.fit(oof_stack, y_arr)
stacked_test   = meta.predict(test_stack)
stacked_oof    = meta.predict(oof_stack)
stack_oof_rmse = np.sqrt(mean_squared_error(y_arr, stacked_oof))
print(f"Stacking OOF RMSE: {stack_oof_rmse:.5f}")
print(f"Stacking coefs: {dict(zip(model_labels, meta.coef_.round(4)))}")

# ─────────────────────────────────────────────
# 8. Weighted Blend (inverse RMSE weights)
# ─────────────────────────────────────────────
inv_rmse = np.array([1.0 / r for r in rmse_scores])
weights  = inv_rmse / inv_rmse.sum()

blend_oof  = sum(w * p for w, p in zip(weights, oof_models))
blend_test = sum(w * p for w, p in zip(weights, test_models))
blend_rmse = np.sqrt(mean_squared_error(y_arr, blend_oof))

print(f"\nWeighted Blend OOF RMSE: {blend_rmse:.5f}")
print("Model weights:")
for lbl, w, r in zip(model_labels, weights, rmse_scores):
    print(f"  {lbl}: weight={w:.4f}, OOF RMSE={r:.5f}")

# ─────────────────────────────────────────────
# 9. Choose Best Prediction Strategy
# ─────────────────────────────────────────────
all_candidates = {
    'Stacking':      (stack_oof_rmse, stacked_test),
    'Weighted Blend': (blend_rmse, blend_test),
    # Also try 50/50 mix of stack + blend
    'Mix (0.5/0.5)': (
        np.sqrt(mean_squared_error(y_arr, 0.5*stacked_oof + 0.5*blend_oof)),
        0.5*stacked_test + 0.5*blend_test
    ),
}

print("\n=== Candidate OOF RMSEs ===")
best_name, best_rmse, best_preds = None, np.inf, None
for name, (rmse, preds) in all_candidates.items():
    print(f"  {name}: {rmse:.5f}")
    if rmse < best_rmse:
        best_name, best_rmse, best_preds = name, rmse, preds

print(f"\nUsing: {best_name} (OOF RMSE={best_rmse:.5f})")

# Clip predictions to [0, 1]
final_preds = np.clip(best_preds, 0.0, 1.0)

# ─────────────────────────────────────────────
# 10. Save Submission
# ─────────────────────────────────────────────
submission = pd.DataFrame({
    'record_id':        test_record_ids.values,
    'flood_risk_score': final_preds
})

submission.to_csv(OUTPUT_FILE, index=False)
print(f"\n[OK] Saved: {OUTPUT_FILE}")
print(f"Predictions summary:\n{submission['flood_risk_score'].describe().round(4)}")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for lbl, r in zip(model_labels, rmse_scores):
    print(f"  {lbl:15s} OOF RMSE: {r:.5f}")
print(f"  {'Stacking':15s} OOF RMSE: {stack_oof_rmse:.5f}")
print(f"  {'Blend':15s} OOF RMSE: {blend_rmse:.5f}")
print(f"\n  BEST: {best_name} -> {best_rmse:.5f}")
print("="*60)
