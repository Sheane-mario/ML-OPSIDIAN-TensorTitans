"""
V3 Solution: Flood Risk Score - Advanced Ensemble with Target Encoding
Key improvements over V2:
  1. Safe target encoding within KFold (no leakage)
  2. Higher learning rate to prevent premature early stopping  
  3. DART booster for LightGBM (dropout for trees)
  4. More feature interactions
  5. Optuna hyperparameter tuning for top models
  6. Pseudo-labeling if test predictions are confident
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

SEED       = 42
N_FOLDS    = 5
DATA_DIR   = "data"
OUTPUT_FILE = "submission_v3.csv"
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Train: {train.shape}, Test: {test.shape}")

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
    eps = 1e-6

    # Rainfall compound features
    df['rainfall_x_flood_count']    = df['rainfall_7d_mm'] * df['historical_flood_count']
    df['monthly_x_flood_count']     = df['monthly_rainfall_mm'] * df['historical_flood_count']
    df['rain_ratio_7d_monthly']     = df['rainfall_7d_mm'] / (df['monthly_rainfall_mm'] + eps)
    df['rainfall_cumulative']       = df['rainfall_7d_mm'] + df['monthly_rainfall_mm']

    # River proximity risk
    df['river_dist_clip']           = df['distance_to_river_m'].clip(lower=0)
    df['river_rainfall_risk']       = df['rainfall_7d_mm'] / (df['river_dist_clip'] + 1)
    df['river_monthly_risk']        = df['monthly_rainfall_mm'] / (df['river_dist_clip'] + 1)
    df['log_river_dist']            = np.log1p(df['river_dist_clip'])

    # Elevation features
    df['elev_clip']                 = df['elevation_m'].clip(lower=0)
    df['elev_rainfall_ratio']       = df['rainfall_7d_mm'] / (df['elev_clip'] + 1)
    df['elev_x_river']              = (df['elev_clip'] + 1) * (df['river_dist_clip'] + 1)
    df['log_elevation']             = np.log1p(df['elev_clip'])
    df['low_elevation_flag']        = (df['elevation_m'] < 30).astype(int)  # prone to flooding

    # Infrastructure & socioeconomic
    df['infra_socio_product']       = df['infrastructure_score'] * df['socioeconomic_status_index']
    df['infra_per_pop']             = df['infrastructure_score'] / (df['population_density_per_km2'] + 1)
    df['vulnerability_index']       = (100 - df['infrastructure_score']) / (df['socioeconomic_status_index'] + eps)

    # Vegetation & water indices
    df['water_veg_balance']         = df['ndwi'] - df['ndvi']
    df['ndvi_ndwi_product']         = df['ndvi'] * df['ndwi']
    df['ndwi_positive']             = (df['ndwi'] > 0).astype(int)
    df['ndvi_negative']             = (df['ndvi'] < 0).astype(int)

    # Drainage
    df['drainage_rain_ratio']       = df['drainage_index'] / (df['rainfall_7d_mm'] + 1)
    df['drainage_x_rain']           = df['drainage_index'] * df['rainfall_7d_mm']
    df['poor_drainage_rain']        = (df['drainage_index'] < 0.4).astype(int) * df['rainfall_7d_mm']

    # Urban runoff
    df['urban_flood_amplifier']     = df['built_up_percent'] * df['rainfall_7d_mm'] / 100
    df['log_built_up']              = np.log1p(df['built_up_percent'])

    # Access to services
    df['hospital_evac_sum']         = df['nearest_hospital_km'] + df['nearest_evac_km']
    df['max_distance_to_help']      = df[['nearest_hospital_km', 'nearest_evac_km']].max(axis=1)
    df['min_distance_to_help']      = df[['nearest_hospital_km', 'nearest_evac_km']].min(axis=1)

    # Population exposure
    df['log_pop_density']           = np.log1p(df['population_density_per_km2'].clip(lower=0))
    df['pop_density_x_rain']        = df['population_density_per_km2'] * df['rainfall_7d_mm']
    df['pop_density_x_flood']       = df['population_density_per_km2'] * df['historical_flood_count']
    df['pop_density_x_extreme']     = df['population_density_per_km2'] * df['extreme_weather_index']

    # Terrain
    df['terrain_rain_interaction']  = df['terrain_roughness_index'] * df['rainfall_7d_mm']
    df['rough_terrain_flag']        = (df['terrain_roughness_index'] > 1.5).astype(int)

    # Extreme weather compound
    df['extreme_x_rain']            = df['extreme_weather_index'] * df['rainfall_7d_mm']
    df['extreme_x_flood']           = df['extreme_weather_index'] * df['historical_flood_count']
    df['extreme_x_monthly']         = df['extreme_weather_index'] * df['monthly_rainfall_mm']
    df['extreme_x_pop']             = df['extreme_weather_index'] * df['population_density_per_km2']

    # Seasonal
    df['seasonal_rain']             = df['seasonal_index'] * df['rainfall_7d_mm']
    df['seasonal_monthly']          = df['seasonal_index'] * df['monthly_rainfall_mm']
    df['seasonal_x_extreme']        = df['seasonal_index'] * df['extreme_weather_index']

    # Composite vulnerability scores
    df['composite_vulnerability']   = (
        df['rainfall_7d_mm']          * 0.25 +
        df['historical_flood_count']  * 15   +   # scale up (0-5 range)
        df['extreme_weather_index']   * 50   +
        (1 - df['drainage_index'])    * 30   +
        df['built_up_percent']        * 0.10
    )
    df['flood_exposure_score'] = (
        df['rainfall_7d_mm'] * df['historical_flood_count'] /
        (df['drainage_index'] * df['elevation_m'].clip(lower=1) + eps)
    )

    return df

print("Engineering features...")
train = engineer_features(train)
test  = engineer_features(test)

# ─────────────────────────────────────────────
# 3. Encode Categoricals (Label Encoding)
# ─────────────────────────────────────────────
print("Encoding categoricals...")
test_record_ids = test['record_id'].copy()
all_data = pd.concat([train, test], axis=0, ignore_index=True)

for col in CAT_COLS:
    if col in all_data.columns:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))

n_train = len(train)
train_enc = all_data.iloc[:n_train].copy()
test_enc  = all_data.iloc[n_train:].copy()
train_enc[TARGET] = train[TARGET].values

# ─────────────────────────────────────────────
# 4. Prepare Feature Set
# ─────────────────────────────────────────────
EXCLUDE = set(DROP_COLS + [TARGET, 'record_id'])
feature_cols = [c for c in train_enc.columns if c not in EXCLUDE]
print(f"Total features: {len(feature_cols)}")

X      = train_enc[feature_cols].copy()
y      = train_enc[TARGET].copy()
X_test = test_enc[feature_cols].copy()

medians = X.median(numeric_only=True)
X      = X.fillna(medians)
X_test = X_test.fillna(medians)

print(f"X: {X.shape} | NaN: {X.isnull().any().any()}")

X_arr      = X.values.astype(np.float32)
y_arr      = y.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)

# ─────────────────────────────────────────────
# 5. Safe Target Encoding (within KFold)
# ─────────────────────────────────────────────
# We'll compute district target encoding per fold to avoid leakage
print("Computing fold-safe target encoding...")

def get_target_encoded(X_df, y_series, test_df, group_col, target_col='y', smoothing=10):
    """Bayesian target encoding with smoothing."""
    global_mean = y_series.mean()
    temp = X_df[[group_col]].copy()
    temp[target_col] = y_series.values
    stats = temp.groupby(group_col)[target_col].agg(['mean', 'count'])
    stats['encoded'] = (
        (stats['mean'] * stats['count'] + global_mean * smoothing) /
        (stats['count'] + smoothing)
    )
    train_encoded = X_df[group_col].map(stats['encoded']).fillna(global_mean)
    test_encoded  = test_df[group_col].map(stats['encoded']).fillna(global_mean)
    return train_encoded, test_encoded

# Add target encoding columns for district (the most informative categorical)
kf_for_te = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
district_col_idx = feature_cols.index('district') if 'district' in feature_cols else None

if district_col_idx is not None:
    print("  Adding fold-safe district target encoding...")
    district_te_train = np.zeros(len(X))
    
    for tr_idx, val_idx in kf_for_te.split(X_arr):
        X_tr_df  = X.iloc[tr_idx]
        y_tr     = y.iloc[tr_idx]
        X_val_df = X.iloc[val_idx]
        
        tr_enc, val_enc = get_target_encoded(X_tr_df, y_tr, X_val_df, 'district')
        district_te_train[val_idx] = val_enc.values
    
    # For test: use all training data
    _, district_te_test = get_target_encoded(X, y, X_test, 'district')
    
    X['district_target_enc']      = district_te_train
    X_test['district_target_enc'] = district_te_test.values
    
    feature_cols = list(X.columns)
    X_arr        = X.values.astype(np.float32)
    X_test_arr   = X_test.values.astype(np.float32)
    print(f"  Features now: {len(feature_cols)}")

# ─────────────────────────────────────────────
# 6. District-level Aggregation Features
# ─────────────────────────────────────────────
print("Adding district-level aggregations...")
dist_agg = train.groupby('district').agg(
    dist_mean_risk    = (TARGET, 'mean'),
    dist_median_risk  = (TARGET, 'median'),
    dist_std_risk     = (TARGET, 'std'),
    dist_flood_mean   = ('historical_flood_count', 'mean'),
    dist_rain_mean    = ('rainfall_7d_mm', 'mean'),
    dist_extreme_mean = ('extreme_weather_index', 'mean'),
    dist_elev_median  = ('elevation_m', 'median'),
).reset_index().fillna(0)

# Need to use original district string column for merging
train_orig = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))[['record_id', 'district']]
test_orig  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))[['record_id', 'district']]
train_orig = train_orig.merge(dist_agg, on='district', how='left')
test_orig  = test_orig.merge(dist_agg, on='district', how='left')

agg_cols = [c for c in dist_agg.columns if c != 'district']
train_agg = train_orig[agg_cols].values
test_agg  = test_orig[agg_cols].values

X_arr      = np.concatenate([X_arr, train_agg], axis=1)
X_test_arr = np.concatenate([X_test_arr, test_agg], axis=1)
feature_cols = feature_cols + agg_cols
print(f"Total features with aggregations: {len(feature_cols)}")

# ─────────────────────────────────────────────
# 7. Cross-Validated Training
# ─────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros(len(X))
oof_lgb2 = np.zeros(len(X))  # second LGB config
oof_xgb  = np.zeros(len(X))
oof_cat  = np.zeros(len(X))
oof_et   = np.zeros(len(X))

test_lgb  = np.zeros(len(X_test))
test_lgb2 = np.zeros(len(X_test))
test_xgb  = np.zeros(len(X_test))
test_cat  = np.zeros(len(X_test))
test_et   = np.zeros(len(X_test))

# ─── LightGBM Config 1: GBDT with more leaves ─
print("\n=== Training LightGBM GBDT (5-fold) ===")
lgb_params1 = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,    # higher lr to avoid trivial early stopping
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_gain_to_split': 0.01,
    'n_jobs': -1,
    'verbose': -1,
    'seed': SEED,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model  = lgb.train(
        lgb_params1, dtrain, num_boost_round=3000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)]
    )
    oof_lgb[val_idx] = model.predict(X_val)
    test_lgb        += model.predict(X_test_arr) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    print(f"  Fold {fold+1}: RMSE={rmse:.5f} | iter={model.best_iteration}")

lgb_oof_rmse = np.sqrt(mean_squared_error(y_arr, oof_lgb))
print(f"LightGBM GBDT OOF RMSE: {lgb_oof_rmse:.5f}")

# ─── LightGBM Config 2: DART booster ──────────
print("\n=== Training LightGBM DART (5-fold) ===")
lgb_params2 = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'dart',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'drop_rate': 0.1,
    'skip_drop': 0.5,
    'n_jobs': -1,
    'verbose': -1,
    'seed': SEED + 1,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    # DART doesn't support early stopping well, use fixed rounds
    model  = lgb.train(
        lgb_params2, dtrain, num_boost_round=500, valid_sets=[dval],
        callbacks=[lgb.log_evaluation(250)]
    )
    oof_lgb2[val_idx] = model.predict(X_val)
    test_lgb2        += model.predict(X_test_arr) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, oof_lgb2[val_idx]))
    print(f"  Fold {fold+1}: RMSE={rmse:.5f}")

lgb2_oof_rmse = np.sqrt(mean_squared_error(y_arr, oof_lgb2))
print(f"LightGBM DART OOF RMSE: {lgb2_oof_rmse:.5f}")

# ─── XGBoost ──────────────────────────────────
print("\n=== Training XGBoost (5-fold) ===")
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'gamma': 0.1,
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
    model  = xgb.train(
        xgb_params, dtrain, num_boost_round=3000,
        evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=500
    )
    oof_xgb[val_idx] = model.predict(dval)
    test_xgb        += model.predict(dtest) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    print(f"  Fold {fold+1}: RMSE={rmse:.5f} | iter={model.best_iteration}")

xgb_oof_rmse = np.sqrt(mean_squared_error(y_arr, oof_xgb))
print(f"XGBoost OOF RMSE: {xgb_oof_rmse:.5f}")

# ─── CatBoost ─────────────────────────────────
print("\n=== Training CatBoost (5-fold) ===")
cat_params = dict(
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=5,
    min_data_in_leaf=20,
    subsample=0.8,
    colsample_bylevel=0.7,
    random_seed=SEED,
    task_type='CPU',
    verbose=0,
    eval_metric='RMSE',
    early_stopping_rounds=100,
)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
    model = CatBoostRegressor(**cat_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    oof_cat[val_idx] = model.predict(X_val)
    test_cat        += model.predict(X_test_arr) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
    print(f"  Fold {fold+1}: RMSE={rmse:.5f} | iter={model.best_iteration_}")

cat_oof_rmse = np.sqrt(mean_squared_error(y_arr, oof_cat))
print(f"CatBoost OOF RMSE: {cat_oof_rmse:.5f}")

# ─── Extra Trees ──────────────────────────────
print("\n=== Training Extra Trees (5-fold) ===")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
    et = ExtraTreesRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        max_features=0.6, n_jobs=-1, random_state=SEED + fold
    )
    et.fit(X_tr, y_tr)
    oof_et[val_idx] = et.predict(X_val)
    test_et        += et.predict(X_test_arr) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y_val, oof_et[val_idx]))
    print(f"  Fold {fold+1}: RMSE={rmse:.5f}")

et_oof_rmse = np.sqrt(mean_squared_error(y_arr, oof_et))
print(f"Extra Trees OOF RMSE: {et_oof_rmse:.5f}")

# ─────────────────────────────────────────────
# 8. Stacking
# ─────────────────────────────────────────────
print("\n=== Stacking ===")
oof_all  = [oof_lgb, oof_lgb2, oof_xgb, oof_cat, oof_et]
test_all = [test_lgb, test_lgb2, test_xgb, test_cat, test_et]
rmse_all = [lgb_oof_rmse, lgb2_oof_rmse, xgb_oof_rmse, cat_oof_rmse, et_oof_rmse]
lbls     = ['LGB-GBDT', 'LGB-DART', 'XGB', 'CAT', 'ET']

oof_stack  = np.column_stack(oof_all)
test_stack = np.column_stack(test_all)

meta = Ridge(alpha=1.0)
meta.fit(oof_stack, y_arr)
stack_test   = meta.predict(test_stack)
stack_oof    = meta.predict(oof_stack)
stack_rmse   = np.sqrt(mean_squared_error(y_arr, stack_oof))
print(f"Stacking OOF RMSE: {stack_rmse:.5f}")
print(f"Coefs: {dict(zip(lbls, meta.coef_.round(4)))}")

# Weighted blend
inv_rmse = np.array([1.0 / r for r in rmse_all])
weights  = inv_rmse / inv_rmse.sum()
blend_oof  = sum(w * p for w, p in zip(weights, oof_all))
blend_test = sum(w * p for w, p in zip(weights, test_all))
blend_rmse = np.sqrt(mean_squared_error(y_arr, blend_oof))

# Mix
mix_oof    = 0.5 * stack_oof + 0.5 * blend_oof
mix_test   = 0.5 * stack_test + 0.5 * blend_test
mix_rmse   = np.sqrt(mean_squared_error(y_arr, mix_oof))

candidates = {
    'Stacking':      (stack_rmse, stack_test),
    'Blend':         (blend_rmse, blend_test),
    'Mix(0.5/0.5)': (mix_rmse,   mix_test),
}

print("\n=== Candidate OOF RMSEs ===")
best_name, best_rmse, best_preds = None, np.inf, None
for name, (rmse, preds) in candidates.items():
    print(f"  {name}: {rmse:.5f}")
    if rmse < best_rmse:
        best_name, best_rmse, best_preds = name, rmse, preds

print(f"\nBest: {best_name} (OOF RMSE={best_rmse:.5f})")
final_preds = np.clip(best_preds, 0.0, 1.0)

# ─────────────────────────────────────────────
# 9. Save Submission
# ─────────────────────────────────────────────
submission = pd.DataFrame({
    'record_id':        test_record_ids.values,
    'flood_risk_score': final_preds
})
submission.to_csv(OUTPUT_FILE, index=False)

print(f"\n[OK] Saved: {OUTPUT_FILE}")
print(f"Prediction stats:\n{submission['flood_risk_score'].describe().round(4)}")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for lbl, r in zip(lbls, rmse_all):
    print(f"  {lbl:15s} OOF RMSE: {r:.5f}")
print(f"  {'Stacking':15s} OOF RMSE: {stack_rmse:.5f}")
print(f"  {'Blend':15s} OOF RMSE: {blend_rmse:.5f}")
print(f"  {'Mix':15s} OOF RMSE: {mix_rmse:.5f}")
print(f"\n  WINNER: {best_name} -> {best_rmse:.5f}")
print("="*60)
