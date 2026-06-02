"""
V8: V7 clean base + fold-safe place TE (NO leaky place stats)

Key insight: between-place variation accounts for ~4% of R^2
(place_mean_risk has 0.195 correlation with target)
Fold-safe place TE captures this without leakage.

The leakage in V5/V6 was the direct merge of place_stats computed from ALL training
onto the training set. Fold-safe TE only uses OTHER folds' data -> no leakage.

Expected R^2 improvement: 3.46% -> 5-8%
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

SEED        = 42
N_FOLDS     = 10
DATA_DIR    = "data"
OUTPUT_FILE = "submission_v8.csv"
np.random.seed(SEED)

print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Train: {train.shape}, Test: {test.shape}")

TARGET = 'flood_risk_score'
test_record_ids = test['record_id'].copy()

# ─────────────────────────────────────────────
# Date Features
# ─────────────────────────────────────────────
for df in [train, test]:
    df['gen_date']        = pd.to_datetime(df['generation_date'])
    df['gen_month']       = df['gen_date'].dt.month
    df['gen_year']        = df['gen_date'].dt.year
    df['gen_day_of_year'] = df['gen_date'].dt.dayofyear
    df['gen_quarter']     = df['gen_date'].dt.quarter
    df['is_ne_monsoon']   = df['gen_month'].isin([12, 1, 2]).astype(int)
    df['is_sw_monsoon']   = df['gen_month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['gen_month_sin']   = np.sin(2 * np.pi * df['gen_month'] / 12)
    df['gen_month_cos']   = np.cos(2 * np.pi * df['gen_month'] / 12)

# Reason / meta flags
for df in [train, test]:
    reason = df['reason_not_good_to_live'].fillna('Other')
    df['reason_flood_flag'] = reason.str.contains('flood', case=False).astype(int)
    df['reason_infra_flag'] = reason.str.contains('infrastructure', case=False).astype(int)
    df['reason_road_flag']  = reason.str.contains('road', case=False).astype(int)
    df['reason_other_flag'] = (reason == 'Other').astype(int)
    df['is_good_binary']    = (df['is_good_to_live'] == 'Yes').astype(int)

# Inundation
for df in [train, test]:
    df['log_inundation']     = np.log1p(df['inundation_area_sqm'])
    df['sqrt_inundation']    = np.sqrt(df['inundation_area_sqm'])
    df['inundation_per_pop'] = df['inundation_area_sqm'] / (df['population_density_per_km2'] + 1)

# record_id numeric (might encode time order)
for df in [train, test]:
    df['record_id_num'] = df['record_id'].str.replace('F', '', regex=False).astype(int)

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    eps = 1e-6

    df['rainfall_x_flood']        = df['rainfall_7d_mm'] * df['historical_flood_count']
    df['monthly_x_flood']         = df['monthly_rainfall_mm'] * df['historical_flood_count']
    df['rain_ratio_7d_monthly']   = df['rainfall_7d_mm'] / (df['monthly_rainfall_mm'] + eps)
    df['rain_cumulative']         = df['rainfall_7d_mm'] + df['monthly_rainfall_mm']

    df['river_clip']              = df['distance_to_river_m'].clip(lower=0)
    df['river_rain_risk']         = df['rainfall_7d_mm'] / (df['river_clip'] + 1)
    df['river_monthly_risk']      = df['monthly_rainfall_mm'] / (df['river_clip'] + 1)
    df['river_x_elevation']       = df['river_clip'] * df['elevation_m'].clip(lower=0)

    df['elev_clip']               = df['elevation_m'].clip(lower=0)
    df['elev_rain_ratio']         = df['rainfall_7d_mm'] / (df['elev_clip'] + 1)
    df['low_elev_flag']           = (df['elevation_m'] < 30).astype(int)

    df['infra_socio']             = df['infrastructure_score'] * df['socioeconomic_status_index']
    df['water_veg_balance']       = df['ndwi'] - df['ndvi']
    df['ndvi_ndwi_product']       = df['ndvi'] * df['ndwi']
    df['ndwi_sq']                 = df['ndwi'] ** 2

    df['drainage_rain_ratio']     = df['drainage_index'] / (df['rainfall_7d_mm'] + 1)
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

    if 'rainfall_7d_mm_log1p' in df.columns:
        df['log_rain_x_flood']    = df['rainfall_7d_mm_log1p'] * df['historical_flood_count']
        df['log_rain_x_extreme']  = df['rainfall_7d_mm_log1p'] * df['extreme_weather_index']
    if 'distance_to_river_m_log1p' in df.columns:
        df['log_river_x_rain']    = df['distance_to_river_m_log1p'] * df['rainfall_7d_mm']
        df['log_river_x_extreme'] = df['distance_to_river_m_log1p'] * df['extreme_weather_index']
    if 'population_density_per_km2_log1p' in df.columns:
        df['log_pop_x_rain']      = df['population_density_per_km2_log1p'] * df['rainfall_7d_mm']
    if 'ndwi_qmap' in df.columns:
        df['ndwi_qmap_x_rain']    = df['ndwi_qmap'] * df['rainfall_7d_mm']
    if 'elevation_m_yeojohnson' in df.columns:
        df['yj_elev_x_rain']      = df['elevation_m_yeojohnson'] * df['rainfall_7d_mm']

    df['inundation_x_rain']       = df['inundation_area_sqm'] * df['rainfall_7d_mm']
    df['log_inundation_x_extreme']= df['log_inundation'] * df['extreme_weather_index']

    df['composite_vuln']          = (
        df['rainfall_7d_mm']         * 0.3 +
        df['historical_flood_count'] * 15.0 +
        df['extreme_weather_index']  * 50.0 +
        (1 - df['drainage_index'])   * 30.0 +
        df['built_up_percent']       * 0.10
    )
    df['flood_exposure'] = (
        df['rainfall_7d_mm'] * (df['historical_flood_count'] + 1) /
        (df['drainage_index'] * df['elev_clip'].clip(lower=1) + eps)
    )
    return df

print("Engineering features...")
train = engineer_features(train)
test  = engineer_features(test)

# ─────────────────────────────────────────────
# Encode Categoricals
# ─────────────────────────────────────────────
CAT_COLS = [
    'district', 'landcover', 'soil_type', 'water_supply',
    'electricity', 'road_quality', 'urban_rural',
    'water_presence_flag', 'flood_occurrence_current_event',
    'is_good_to_live', 'reason_not_good_to_live', 'place_name'
]
DROP_COLS = ['record_id', 'gen_date', 'generation_date', 'is_synthetic', TARGET]

all_data = pd.concat([train, test], axis=0, ignore_index=True)
for col in CAT_COLS:
    if col in all_data.columns:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))

n_train   = len(train)
train_enc = all_data.iloc[:n_train].copy()
test_enc  = all_data.iloc[n_train:].copy()
train_enc[TARGET] = train[TARGET].values

EXCLUDE      = set(DROP_COLS + [TARGET, 'record_id'])
feature_cols = [c for c in train_enc.columns if c not in EXCLUDE]

X      = train_enc[feature_cols].copy()
y      = train_enc[TARGET].copy()
X_test = test_enc[feature_cols].copy()

medians = X.median(numeric_only=True)
X       = X.fillna(medians)
X_test  = X_test.fillna(medians)

print(f"Base features: {len(feature_cols)}")

# ─────────────────────────────────────────────
# Fold-safe Target Encodings
# CRITICAL: NO separate place_stats merge (that was the V5 leakage)
# ONLY fold-safe Bayesian TE for place and district
# ─────────────────────────────────────────────
def fold_safe_te(X_df, y_series, X_test_df, group_col, smooth, global_mean, n_folds, seed):
    train_te = np.zeros(len(X_df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(X_df):
        col_tr = X_df.iloc[tr_idx][group_col]
        y_tr   = y_series.iloc[tr_idx]
        col_val= X_df.iloc[val_idx][group_col]
        stats  = pd.DataFrame({'k': col_tr.values, 'y': y_tr.values}).groupby('k')['y'].agg(['mean','count'])
        stats['enc'] = (stats['mean']*stats['count'] + global_mean*smooth) / (stats['count']+smooth)
        train_te[val_idx] = col_val.map(stats['enc']).fillna(global_mean).values
    # Test: full-data stats
    all_stats = pd.DataFrame({'k': X_df[group_col].values, 'y': y_series.values}).groupby('k')['y'].agg(['mean','count'])
    all_stats['enc'] = (all_stats['mean']*all_stats['count'] + global_mean*smooth) / (all_stats['count']+smooth)
    test_te = X_test_df[group_col].map(all_stats['enc']).fillna(global_mean).values
    return train_te, test_te

global_mean = y.mean()
print("Computing fold-safe target encodings (place + district)...")

# Place TE: smooth=5 (26 rows per place avg → low smoothing is fine)
# Between-place variation: place_mean has 0.195 corr with target -> key signal!
place_te_tr, place_te_te = fold_safe_te(X, y, X_test, 'place_name',
                                         smooth=5, global_mean=global_mean,
                                         n_folds=N_FOLDS, seed=SEED)
# District TE: smooth=30 (25 districts → high smoothing)
dist_te_tr, dist_te_te   = fold_safe_te(X, y, X_test, 'district',
                                         smooth=30, global_mean=global_mean,
                                         n_folds=N_FOLDS, seed=SEED)
# Soil type TE: smooth=15
soil_te_tr, soil_te_te   = fold_safe_te(X, y, X_test, 'soil_type',
                                         smooth=15, global_mean=global_mean,
                                         n_folds=N_FOLDS, seed=SEED)
# Landcover TE: smooth=15
lc_te_tr, lc_te_te       = fold_safe_te(X, y, X_test, 'landcover',
                                         smooth=15, global_mean=global_mean,
                                         n_folds=N_FOLDS, seed=SEED)

# Validate: corr of place TE with target (train)
print(f"Place TE train corr with target: {np.corrcoef(place_te_tr, y.values)[0,1]:.4f}")
print(f"District TE train corr with target: {np.corrcoef(dist_te_tr, y.values)[0,1]:.4f}")

X['place_te']    = place_te_tr
X['district_te'] = dist_te_tr
X['soil_te']     = soil_te_tr
X['landcover_te']= lc_te_tr
X_test['place_te']    = place_te_te
X_test['district_te'] = dist_te_te
X_test['soil_te']     = soil_te_te
X_test['landcover_te']= lc_te_te

feature_cols = list(X.columns)

X_arr      = X.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)
y_arr      = y.values.astype(np.float32)
print(f"Final features: {len(feature_cols)} | NaN: {np.isnan(X_arr).any()}")

# ─────────────────────────────────────────────
# Train Models (10-fold)
# ─────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = {}
test_preds = {}

# LightGBM (MAE)
print("\n=== LightGBM (MAE obj) ===")
lgb_params = {
    'objective': 'regression_l1', 'metric': 'rmse',
    'learning_rate': 0.03, 'num_leaves': 63,
    'min_child_samples': 20, 'feature_fraction': 0.7,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'n_jobs': -1, 'verbose': -1, 'seed': SEED,
}
oof['lgb'] = np.zeros(len(X))
test_preds['lgb'] = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    dt = lgb.Dataset(Xtr, label=ytr, feature_name=feature_cols)
    dv = lgb.Dataset(Xvl, label=yvl, reference=dt)
    m  = lgb.train(lgb_params, dt, 5000, valid_sets=[dv],
                   callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(9999)])
    oof['lgb'][val_idx] = m.predict(Xvl)
    test_preds['lgb']  += m.predict(X_test_arr) / N_FOLDS
    print(f"  Fold {fold+1:2d}: RMSE={np.sqrt(mean_squared_error(yvl, oof['lgb'][val_idx])):.5f} R2={r2_score(yvl, oof['lgb'][val_idx]):.4f} | iter={m.best_iteration}")
r = np.sqrt(mean_squared_error(y_arr, oof['lgb']))
print(f"LGB OOF RMSE={r:.5f}, R2={r2_score(y_arr, oof['lgb']):.4f}")

# LightGBM (RMSE, larger model)
print("\n=== LightGBM (RMSE, 127 leaves) ===")
lgb_params2 = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.03, 'num_leaves': 127,
    'min_child_samples': 15, 'feature_fraction': 0.75,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'reg_alpha': 0.05, 'reg_lambda': 0.5,
    'n_jobs': -1, 'verbose': -1, 'seed': SEED+1,
}
oof['lgb2'] = np.zeros(len(X))
test_preds['lgb2'] = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    dt = lgb.Dataset(Xtr, label=ytr, feature_name=feature_cols)
    dv = lgb.Dataset(Xvl, label=yvl, reference=dt)
    m  = lgb.train(lgb_params2, dt, 5000, valid_sets=[dv],
                   callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(9999)])
    oof['lgb2'][val_idx] = m.predict(Xvl)
    test_preds['lgb2']  += m.predict(X_test_arr) / N_FOLDS
    print(f"  Fold {fold+1:2d}: RMSE={np.sqrt(mean_squared_error(yvl, oof['lgb2'][val_idx])):.5f} R2={r2_score(yvl, oof['lgb2'][val_idx]):.4f} | iter={m.best_iteration}")
r = np.sqrt(mean_squared_error(y_arr, oof['lgb2']))
print(f"LGB2 OOF RMSE={r:.5f}, R2={r2_score(y_arr, oof['lgb2']):.4f}")

# XGBoost
print("\n=== XGBoost ===")
xgb_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'learning_rate': 0.03, 'max_depth': 7,
    'min_child_weight': 10, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'gamma': 0.1,
    'n_jobs': -1, 'seed': SEED, 'tree_method': 'hist', 'verbosity': 0,
}
oof['xgb'] = np.zeros(len(X))
test_preds['xgb'] = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    dt = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_cols)
    dv = xgb.DMatrix(Xvl, label=yvl, feature_names=feature_cols)
    ds = xgb.DMatrix(X_test_arr, feature_names=feature_cols)
    m  = xgb.train(xgb_params, dt, 5000, evals=[(dv,'val')],
                   early_stopping_rounds=150, verbose_eval=9999)
    oof['xgb'][val_idx] = m.predict(dv)
    test_preds['xgb']  += m.predict(ds) / N_FOLDS
    print(f"  Fold {fold+1:2d}: RMSE={np.sqrt(mean_squared_error(yvl, oof['xgb'][val_idx])):.5f} R2={r2_score(yvl, oof['xgb'][val_idx]):.4f} | iter={m.best_iteration}")
r = np.sqrt(mean_squared_error(y_arr, oof['xgb']))
print(f"XGB OOF RMSE={r:.5f}, R2={r2_score(y_arr, oof['xgb']):.4f}")

# CatBoost
print("\n=== CatBoost ===")
cat_params = dict(
    iterations=5000, learning_rate=0.03, depth=8,
    l2_leaf_reg=5, min_data_in_leaf=20,
    subsample=0.8, colsample_bylevel=0.7,
    random_seed=SEED, task_type='CPU', verbose=0,
    eval_metric='RMSE', early_stopping_rounds=150,
)
oof['cat'] = np.zeros(len(X))
test_preds['cat'] = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    m = CatBoostRegressor(**cat_params)
    m.fit(Xtr, ytr, eval_set=(Xvl, yvl), use_best_model=True)
    oof['cat'][val_idx] = m.predict(Xvl)
    test_preds['cat']  += m.predict(X_test_arr) / N_FOLDS
    print(f"  Fold {fold+1:2d}: RMSE={np.sqrt(mean_squared_error(yvl, oof['cat'][val_idx])):.5f} R2={r2_score(yvl, oof['cat'][val_idx]):.4f} | iter={m.best_iteration_}")
r = np.sqrt(mean_squared_error(y_arr, oof['cat']))
print(f"CAT OOF RMSE={r:.5f}, R2={r2_score(y_arr, oof['cat']):.4f}")

# ExtraTrees
print("\n=== Extra Trees ===")
oof['et'] = np.zeros(len(X))
test_preds['et'] = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    m = ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_leaf=5,
                             max_features=0.6, n_jobs=-1, random_state=SEED+fold)
    m.fit(Xtr, ytr)
    oof['et'][val_idx] = m.predict(Xvl)
    test_preds['et']  += m.predict(X_test_arr) / N_FOLDS
    print(f"  Fold {fold+1:2d}: RMSE={np.sqrt(mean_squared_error(yvl, oof['et'][val_idx])):.5f} R2={r2_score(yvl, oof['et'][val_idx]):.4f}")
r = np.sqrt(mean_squared_error(y_arr, oof['et']))
print(f"ET OOF RMSE={r:.5f}, R2={r2_score(y_arr, oof['et']):.4f}")

# ─────────────────────────────────────────────
# Stacking
# ─────────────────────────────────────────────
print("\n=== Stacking ===")
model_names = list(oof.keys())
oof_stack   = np.column_stack([oof[k] for k in model_names])
test_stack  = np.column_stack([test_preds[k] for k in model_names])
rmse_list   = [np.sqrt(mean_squared_error(y_arr, oof[k])) for k in model_names]
r2_list     = [r2_score(y_arr, oof[k]) for k in model_names]

ridge = Ridge(alpha=1.0)
ridge.fit(oof_stack, y_arr)
stack_oof   = ridge.predict(oof_stack)
stack_test  = ridge.predict(test_stack)
stack_rmse  = np.sqrt(mean_squared_error(y_arr, stack_oof))
stack_r2    = r2_score(y_arr, stack_oof)

inv_rmse    = np.array([1.0/r for r in rmse_list])
weights     = inv_rmse / inv_rmse.sum()
blend_oof   = sum(w*oof[k] for w,k in zip(weights, model_names))
blend_test  = sum(w*test_preds[k] for w,k in zip(weights, model_names))
blend_rmse  = np.sqrt(mean_squared_error(y_arr, blend_oof))
blend_r2    = r2_score(y_arr, blend_oof)

print("\n=== Final Results ===")
for k, r, r2 in zip(model_names, rmse_list, r2_list):
    print(f"  {k:10s} RMSE={r:.5f}, R2={r2:.4f}, pred_std={np.std(test_preds[k]):.4f}")
print(f"  {'Stack':10s} RMSE={stack_rmse:.5f}, R2={stack_r2:.4f}, pred_std={np.std(stack_test):.4f}")
print(f"  {'Blend':10s} RMSE={blend_rmse:.5f}, R2={blend_r2:.4f}, pred_std={np.std(blend_test):.4f}")
print(f"  Ridge coefs: {dict(zip(model_names, ridge.coef_.round(4)))}")

# Best candidate
best_rmse_model = min(zip(model_names + ['stack','blend'],
                          rmse_list + [stack_rmse, blend_rmse]),
                      key=lambda x: x[1])
print(f"\nBest: {best_rmse_model[0]} -> {best_rmse_model[1]:.5f}")

# Save
pd.DataFrame({'record_id': test_record_ids.values,
              'flood_risk_score': np.clip(stack_test, 0, 1)}).to_csv(OUTPUT_FILE, index=False)
pd.DataFrame({'record_id': test_record_ids.values,
              'flood_risk_score': np.clip(blend_test, 0, 1)}).to_csv("submission_v8_blend.csv", index=False)
pd.DataFrame({'record_id': test_record_ids.values,
              'flood_risk_score': np.clip(test_preds['cat'], 0, 1)}).to_csv("submission_v8_cat.csv", index=False)

print(f"\n[OK] Saved: {OUTPUT_FILE}")
print("\n" + "="*60)
print("FINAL SUMMARY V8 (fold-safe place TE, no leakage)")
print("="*60)
for k, r, r2 in zip(model_names, rmse_list, r2_list):
    print(f"  {k:10s} RMSE={r:.5f}, R2={r2:.4f}")
print(f"  {'Stack':10s} RMSE={stack_rmse:.5f}, R2={stack_r2:.4f}")
print(f"  {'Blend':10s} RMSE={blend_rmse:.5f}, R2={blend_r2:.4f}")
print(f"  V7 Stack:  RMSE=0.23459, R2=0.0346")
print(f"  V2 best (LB 0.38363): RMSE=0.23497")
print("="*60)
