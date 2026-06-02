"""
V17 Genesis Ensemble

Goal: Create the ultimate standalone ensemble for blending. 
We observed the "OOF Overfitting Paradox": overly tuning for local OOF (like V16's strict shallow trees) 
caused a loss in Public LB. The LB explicitly rewards models that track high variance fluctuations.
V15 succeeded because it included deep trees (for variance) but clamped them with a Huber stacker (for outlier safety).

V17 Strategy:
- 115 features (Proven best).
- N_FOLDS = 20 for extreme out-of-fold robustness.
- Full spectrum of CatBoost Depths: 3, 4, 5 (baseline) + 8, 10, 12 (high variance tracking).
- Two LGB models: Standard MAE and 255-leaf RMSE (deep).
- Huber Stack to fuse them robustly.
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from catboost import CatBoostRegressor

SEED        = 42
N_FOLDS     = 20   # Extreme robustness
DATA_DIR    = "data"
OUTPUT_FILE = "submission_v17.csv"
np.random.seed(SEED)

print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Train: {train.shape}, Test: {test.shape}")

TARGET = 'flood_risk_score'
test_record_ids = test['record_id'].copy()

# ─── Date / Meta Features ─────────────────────
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
    reason = df['reason_not_good_to_live'].fillna('Other')
    df['reason_flood_flag'] = reason.str.contains('flood', case=False).astype(int)
    df['reason_infra_flag'] = reason.str.contains('infrastructure', case=False).astype(int)
    df['reason_road_flag']  = reason.str.contains('road', case=False).astype(int)
    df['reason_other_flag'] = (reason == 'Other').astype(int)
    df['is_good_binary']    = (df['is_good_to_live'] == 'Yes').astype(int)
    df['log_inundation']    = np.log1p(df['inundation_area_sqm'])
    df['sqrt_inundation']   = np.sqrt(df['inundation_area_sqm'])
    df['inundation_per_pop']= df['inundation_area_sqm'] / (df['population_density_per_km2'] + 1)
    df['record_id_num']     = df['record_id'].str.replace('F', '', regex=False).astype(int)

def engineer_features(df):
    df = df.copy(); eps = 1e-6
    df['rainfall_x_flood']       = df['rainfall_7d_mm'] * df['historical_flood_count']
    df['monthly_x_flood']        = df['monthly_rainfall_mm'] * df['historical_flood_count']
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
        df['rainfall_7d_mm'] * 0.3 + df['historical_flood_count'] * 15.0 +
        df['extreme_weather_index'] * 50.0 + (1-df['drainage_index']) * 30.0 +
        df['built_up_percent'] * 0.10)
    return df

print("Engineering features...")
train = engineer_features(train)
test  = engineer_features(test)

CAT_COLS = ['district','landcover','soil_type','water_supply','electricity',
            'road_quality','urban_rural','water_presence_flag',
            'flood_occurrence_current_event','is_good_to_live',
            'reason_not_good_to_live','place_name']
DROP_COLS = ['record_id','gen_date','generation_date','is_synthetic',TARGET]

all_data = pd.concat([train, test], axis=0, ignore_index=True)
for col in CAT_COLS:
    if col in all_data.columns:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))

n_train   = len(train)
train_enc = all_data.iloc[:n_train].copy()
test_enc  = all_data.iloc[n_train:].copy()
train_enc[TARGET] = train[TARGET].values

EXCLUDE      = set(DROP_COLS + [TARGET])
feature_cols = [c for c in train_enc.columns if c not in EXCLUDE]
X      = train_enc[feature_cols].copy()
y      = train_enc[TARGET].copy()
X_test = test_enc[feature_cols].copy()
medians = X.median(numeric_only=True)
X       = X.fillna(medians)
X_test  = X_test.fillna(medians)

# N_FOLDS parameter is now dynamic to support 20-fold CV
def fold_safe_te(X_df, y_s, X_te_df, col, smooth, gm, n_folds=N_FOLDS, seed=SEED):
    tr_te = np.zeros(len(X_df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(X_df):
        s = pd.DataFrame({'k': X_df.iloc[tr_idx][col].values, 'y': y_s.iloc[tr_idx].values}).groupby('k')['y'].agg(['mean','count'])
        s['enc'] = (s['mean']*s['count'] + gm*smooth) / (s['count']+smooth)
        tr_te[val_idx] = X_df.iloc[val_idx][col].map(s['enc']).fillna(gm).values
    s_all = pd.DataFrame({'k': X_df[col].values, 'y': y_s.values}).groupby('k')['y'].agg(['mean','count'])
    s_all['enc'] = (s_all['mean']*s_all['count'] + gm*smooth) / (s_all['count']+smooth)
    return tr_te, X_te_df[col].map(s_all['enc']).fillna(gm).values

gm = y.mean()
print(f"Computing fold-safe TEs + interactions ({N_FOLDS} folds)...")
te_store = {}
for col, smooth in [('place_name',5),('district',10),('soil_type',10),
                    ('landcover',10),('road_quality',10),('flood_occurrence_current_event',10)]:
    tr_te, te_te = fold_safe_te(X, y, X_test, col, smooth, gm, N_FOLDS, SEED)
    X[f'{col}_te'] = tr_te; X_test[f'{col}_te'] = te_te
    te_store[col]  = (tr_te, te_te)

d_tr, d_te = te_store['district']
f_tr, f_te = te_store['flood_occurrence_current_event']
s_tr, s_te = te_store['soil_type']

for feat, tr_v, te_v, col_name in [
    ('rainfall_7d_mm',       d_tr, d_te, 'dist_te_x_rain'),
    ('extreme_weather_index',d_tr, d_te, 'dist_te_x_extreme'),
    ('historical_flood_count',d_tr,d_te,'dist_te_x_flood'),
    ('rainfall_7d_mm',       f_tr, f_te, 'fl_te_x_rain'),
    ('extreme_weather_index',f_tr, f_te, 'fl_te_x_extreme'),
    ('drainage_index',       d_tr, d_te, 'dist_te_x_drainage'),
    ('river_clip',           d_tr, d_te, 'dist_te_x_river'),
    ('elevation_m',          d_tr, d_te, 'dist_te_x_elev'),
    ('historical_flood_count',f_tr,f_te,'fl_te_x_flood'),
    ('drainage_index',       f_tr, f_te, 'fl_te_x_drainage'),
    ('rainfall_7d_mm',       s_tr, s_te, 'soil_te_x_rain'),
    ('extreme_weather_index',s_tr, s_te, 'soil_te_x_extreme'),
]:
    X[col_name]      = tr_v * X[feat]
    X_test[col_name] = te_v * X_test[feat]

feature_cols = list(X.columns)
X_arr      = X.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)
y_arr      = y.values.astype(np.float32)
print(f"Final features: {len(feature_cols)}")

# ─── Train ───────────────────────────────────
kf  = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = {}; test_preds = {}

# LGB MAE
print("\n=== LGB (MAE) ===")
oof['lgb'] = np.zeros(len(X)); test_preds['lgb'] = np.zeros(len(X_test))
lgb_p = {'objective':'regression_l1','metric':'rmse','learning_rate':0.03,
          'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,
          'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0,
          'n_jobs':-1,'verbose':-1,'seed':SEED}
for fold,(tr_i,val_i) in enumerate(kf.split(X_arr)):
    m = lgb.train(lgb_p, lgb.Dataset(X_arr[tr_i],y_arr[tr_i],feature_name=feature_cols),
                  5000, valid_sets=[lgb.Dataset(X_arr[val_i],y_arr[val_i])],
                  callbacks=[lgb.early_stopping(150,verbose=False)])
    oof['lgb'][val_i] = m.predict(X_arr[val_i]); test_preds['lgb'] += m.predict(X_test_arr)/N_FOLDS

# LGB Deep (255 leaves) for Variance Tracking
print("\n=== LGB (255 leaves) ===")
oof['lgb_big'] = np.zeros(len(X)); test_preds['lgb_big'] = np.zeros(len(X_test))
lgb_big_p = {'objective':'regression','metric':'rmse','learning_rate':0.02,
              'num_leaves':255,'min_child_samples':30,'feature_fraction':0.6,
              'bagging_fraction':0.75,'bagging_freq':5,'reg_alpha':0.2,'reg_lambda':2.0,
              'n_jobs':-1,'verbose':-1,'seed':SEED+5}
for fold,(tr_i,val_i) in enumerate(kf.split(X_arr)):
    m = lgb.train(lgb_big_p, lgb.Dataset(X_arr[tr_i],y_arr[tr_i],feature_name=feature_cols),
                  5000, valid_sets=[lgb.Dataset(X_arr[val_i],y_arr[val_i])],
                  callbacks=[lgb.early_stopping(150,verbose=False)])
    oof['lgb_big'][val_i] = m.predict(X_arr[val_i]); test_preds['lgb_big'] += m.predict(X_test_arr)/N_FOLDS

# CatBoost Full Spectrum: 3, 4, 5 (for stability) + 8, 10, 12 (for variance)
cat_configs = [
    ('cat3',  3,  1.0, 5,   SEED),
    ('cat4',  4,  1.5, 8,   SEED+1),
    ('cat5',  5,  2.0, 10,  SEED+2),
    ('cat8',  8,  5.0, 20,  SEED+3),
    ('cat10', 10, 8.0, 25,  SEED+4),
    ('cat12', 12, 10.0, 30, SEED+5), # Extremely deep to track high-variance fluctuations
]

for name, depth, l2, min_leaf, seed in cat_configs:
    print(f"\n=== CatBoost {name} ===")
    oof[name] = np.zeros(len(X)); test_preds[name] = np.zeros(len(X_test))
    params = dict(iterations=5000, learning_rate=0.03, depth=depth,
                  l2_leaf_reg=l2, min_data_in_leaf=min_leaf,
                  subsample=0.8, colsample_bylevel=0.7,
                  random_seed=seed, task_type='CPU', verbose=0,
                  eval_metric='RMSE', early_stopping_rounds=150)
    for fold,(tr_i,val_i) in enumerate(kf.split(X_arr)):
        m = CatBoostRegressor(**params)
        m.fit(X_arr[tr_i],y_arr[tr_i],eval_set=(X_arr[val_i],y_arr[val_i]),use_best_model=True)
        oof[name][val_i] = m.predict(X_arr[val_i]); test_preds[name] += m.predict(X_test_arr)/N_FOLDS
        print(f"  F{fold+1}: RMSE={np.sqrt(mean_squared_error(y_arr[val_i],oof[name][val_i])):.5f}")

# ─── Stack ───────────────────────────────────
print("\n=== Robust Stacking ===")
model_names = list(oof.keys())
oof_stack   = np.column_stack([oof[k] for k in model_names])
test_stack  = np.column_stack([test_preds[k] for k in model_names])

# Huber Regressor (Epsilon=1.35, Standard robust)
h135 = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000)
h135.fit(oof_stack, y_arr)
h135_oof  = h135.predict(oof_stack)
h135_test = h135.predict(test_stack)
h135_rmse = np.sqrt(mean_squared_error(y_arr, h135_oof))

print("\n=== Final Results ===")
for i, k in enumerate(model_names):
    r = np.sqrt(mean_squared_error(y_arr, oof[k]))
    print(f"  {k:10s} RMSE={r:.5f}  H135={h135.coef_[i]:.4f}")

print(f"\n  {'H135 Stack':10s} RMSE={h135_rmse:.5f} R2={r2_score(y_arr, h135_oof):.4f}")

pd.DataFrame({'record_id': test_record_ids.values,
              'flood_risk_score': np.clip(h135_test,0,1)}).to_csv(OUTPUT_FILE,index=False)
print(f"[OK] {OUTPUT_FILE}")
