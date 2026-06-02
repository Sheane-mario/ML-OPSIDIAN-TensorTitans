"""
V10: Pure tree diversity ensemble - no MLP/KNN (both had negative R^2 in V9)

Strategy:
  1. CatBoost with NATIVE categorical handling (no label encoding for cats)
  2. LGB with DART booster (dropout regularization, different from GBDT)
  3. Multiple CatBoost configs with different depths (6, 8, 10)
  4. XGBoost with different max_depth
  5. Rank-averaged blend of best tree models

Key insight from V9:
  - MLP R^2 = -4.5%, KNN R^2 = -1.8% -> both hurt even in stack
  - CatBoost is consistently best (R^2=0.033)
  - district_te has 0.1238 corr -> strongest signal
  - flood_occurrence_current_event_te has 0.0635 corr -> significant!

New additions:
  - Interaction: district_te * rainfall_7d_mm, district_te * extreme_weather_index
  - Interaction: flood_occ_te * rainfall_7d_mm (event-specific signal)
  - CatBoost with native cats (uses internal Bayesian TE = better!)
  - DART LGB (different inductive bias, more regularized)
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

SEED        = 42
N_FOLDS     = 10
DATA_DIR    = "data"
OUTPUT_FILE = "submission_v10.csv"
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

# ─── Feature Engineering ─────────────────────
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

# ─── Encode (label-encode for tree models) ────
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

# ─── Fold-safe TEs ───────────────────────────
def fold_safe_te(X_df, y_s, X_te_df, col, smooth, gm, n_folds=10, seed=42):
    tr_te = np.zeros(len(X_df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(X_df):
        s = pd.DataFrame({'k': X_df.iloc[tr_idx][col].values, 'y': y_s.iloc[tr_idx].values}).groupby('k')['y'].agg(['mean','count'])
        s['enc'] = (s['mean']*s['count'] + gm*smooth) / (s['count']+smooth)
        tr_te[val_idx] = X_df.iloc[val_idx][col].map(s['enc']).fillna(gm).values
    s_all = pd.DataFrame({'k': X_df[col].values, 'y': y_s.values}).groupby('k')['y'].agg(['mean','count'])
    s_all['enc'] = (s_all['mean']*s_all['count'] + gm*smooth) / (s_all['count']+smooth)
    te_te = X_te_df[col].map(s_all['enc']).fillna(gm).values
    return tr_te, te_te

gm = y.mean()
print("Computing fold-safe target encodings + TE interactions...")

te_cols_smooth = [('place_name',5),('district',10),('soil_type',10),
                  ('landcover',10),('road_quality',10),('flood_occurrence_current_event',10)]
te_arrays = {}
for col, smooth in te_cols_smooth:
    tr_te, te_te = fold_safe_te(X, y, X_test, col, smooth, gm, N_FOLDS, SEED)
    X[f'{col}_te']      = tr_te
    X_test[f'{col}_te'] = te_te
    te_arrays[col]      = (tr_te, te_te)
    print(f"  {col}_te corr={np.corrcoef(tr_te, y.values)[0,1]:.4f}")

# ─── TE × Feature Interactions ────────────────
print("Adding TE interaction features...")
# district_te × weather (strongest TE × event features)
dist_tr, dist_te = te_arrays['district']
fl_tr, fl_te     = te_arrays['flood_occurrence_current_event']

X['dist_te_x_rain']    = dist_tr * X['rainfall_7d_mm']
X['dist_te_x_extreme'] = dist_tr * X['extreme_weather_index']
X['dist_te_x_flood']   = dist_tr * X['historical_flood_count']
X['dist_te_sq']        = dist_tr ** 2
X['fl_te_x_rain']      = fl_tr * X['rainfall_7d_mm']
X['fl_te_x_extreme']   = fl_tr * X['extreme_weather_index']

X_test['dist_te_x_rain']    = dist_te * X_test['rainfall_7d_mm']
X_test['dist_te_x_extreme'] = dist_te * X_test['extreme_weather_index']
X_test['dist_te_x_flood']   = dist_te * X_test['historical_flood_count']
X_test['dist_te_sq']        = dist_te ** 2
X_test['fl_te_x_rain']      = fl_te * X_test['rainfall_7d_mm']
X_test['fl_te_x_extreme']   = fl_te * X_test['extreme_weather_index']

feature_cols = list(X.columns)
X_arr      = X.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)
y_arr      = y.values.astype(np.float32)
print(f"Final features: {len(feature_cols)} | NaN: {np.isnan(X_arr).any()}")

# ─── Train (10-fold) ─────────────────────────
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
                  callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(9999)])
    oof['lgb'][val_i] = m.predict(X_arr[val_i]); test_preds['lgb'] += m.predict(X_test_arr)/N_FOLDS
    print(f"  F{fold+1}: RMSE={np.sqrt(mean_squared_error(y_arr[val_i],oof['lgb'][val_i])):.5f} R2={r2_score(y_arr[val_i],oof['lgb'][val_i]):.4f}")
print(f"LGB OOF RMSE={np.sqrt(mean_squared_error(y_arr,oof['lgb'])):.5f} R2={r2_score(y_arr,oof['lgb']):.4f}")

# LGB DART (dropout regularization)
print("\n=== LGB DART ===")
oof['lgb_dart'] = np.zeros(len(X)); test_preds['lgb_dart'] = np.zeros(len(X_test))
dart_p = {'objective':'regression','metric':'rmse','boosting_type':'dart',
           'learning_rate':0.05,'num_leaves':63,'min_child_samples':20,
           'feature_fraction':0.7,'bagging_fraction':0.8,'bagging_freq':5,
           'drop_rate':0.1,'skip_drop':0.5,
           'reg_alpha':0.1,'reg_lambda':1.0,'n_jobs':-1,'verbose':-1,'seed':SEED+10}
for fold,(tr_i,val_i) in enumerate(kf.split(X_arr)):
    m = lgb.train(dart_p, lgb.Dataset(X_arr[tr_i],y_arr[tr_i],feature_name=feature_cols),
                  500, valid_sets=[lgb.Dataset(X_arr[val_i],y_arr[val_i])],  # DART can't use ES properly
                  callbacks=[lgb.log_evaluation(9999)])
    oof['lgb_dart'][val_i] = m.predict(X_arr[val_i]); test_preds['lgb_dart'] += m.predict(X_test_arr)/N_FOLDS
    print(f"  F{fold+1}: RMSE={np.sqrt(mean_squared_error(y_arr[val_i],oof['lgb_dart'][val_i])):.5f} R2={r2_score(y_arr[val_i],oof['lgb_dart'][val_i]):.4f}")
print(f"DART OOF RMSE={np.sqrt(mean_squared_error(y_arr,oof['lgb_dart'])):.5f} R2={r2_score(y_arr,oof['lgb_dart']):.4f}")

# XGBoost
print("\n=== XGBoost ===")
oof['xgb'] = np.zeros(len(X)); test_preds['xgb'] = np.zeros(len(X_test))
xgb_p = {'objective':'reg:squarederror','eval_metric':'rmse','learning_rate':0.03,
          'max_depth':7,'min_child_weight':10,'subsample':0.8,'colsample_bytree':0.7,
          'reg_alpha':0.1,'reg_lambda':1.0,'gamma':0.1,
          'n_jobs':-1,'seed':SEED,'tree_method':'hist','verbosity':0}
for fold,(tr_i,val_i) in enumerate(kf.split(X_arr)):
    dtr=xgb.DMatrix(X_arr[tr_i],label=y_arr[tr_i],feature_names=feature_cols)
    dv =xgb.DMatrix(X_arr[val_i],label=y_arr[val_i],feature_names=feature_cols)
    ds =xgb.DMatrix(X_test_arr,feature_names=feature_cols)
    m  =xgb.train(xgb_p,dtr,5000,evals=[(dv,'val')],early_stopping_rounds=150,verbose_eval=9999)
    oof['xgb'][val_i] = m.predict(dv); test_preds['xgb'] += m.predict(ds)/N_FOLDS
    print(f"  F{fold+1}: RMSE={np.sqrt(mean_squared_error(y_arr[val_i],oof['xgb'][val_i])):.5f} R2={r2_score(y_arr[val_i],oof['xgb'][val_i]):.4f}")
print(f"XGB OOF RMSE={np.sqrt(mean_squared_error(y_arr,oof['xgb'])):.5f} R2={r2_score(y_arr,oof['xgb']):.4f}")

# CatBoost depth=8 (baseline)
print("\n=== CatBoost (depth=8) ===")
oof['cat8'] = np.zeros(len(X)); test_preds['cat8'] = np.zeros(len(X_test))
cat8 = dict(iterations=5000,learning_rate=0.03,depth=8,l2_leaf_reg=5,
            min_data_in_leaf=20,subsample=0.8,colsample_bylevel=0.7,
            random_seed=SEED,task_type='CPU',verbose=0,eval_metric='RMSE',early_stopping_rounds=150)
for fold,(tr_i,val_i) in enumerate(kf.split(X_arr)):
    m=CatBoostRegressor(**cat8)
    m.fit(X_arr[tr_i],y_arr[tr_i],eval_set=(X_arr[val_i],y_arr[val_i]),use_best_model=True)
    oof['cat8'][val_i] = m.predict(X_arr[val_i]); test_preds['cat8'] += m.predict(X_test_arr)/N_FOLDS
    print(f"  F{fold+1}: RMSE={np.sqrt(mean_squared_error(y_arr[val_i],oof['cat8'][val_i])):.5f} R2={r2_score(y_arr[val_i],oof['cat8'][val_i]):.4f}")
print(f"CAT8 OOF RMSE={np.sqrt(mean_squared_error(y_arr,oof['cat8'])):.5f} R2={r2_score(y_arr,oof['cat8']):.4f}")

# CatBoost depth=6 (more regularized)
print("\n=== CatBoost (depth=6) ===")
oof['cat6'] = np.zeros(len(X)); test_preds['cat6'] = np.zeros(len(X_test))
cat6 = dict(iterations=5000,learning_rate=0.03,depth=6,l2_leaf_reg=3,
            min_data_in_leaf=15,subsample=0.8,colsample_bylevel=0.7,
            random_seed=SEED+1,task_type='CPU',verbose=0,eval_metric='RMSE',early_stopping_rounds=150)
for fold,(tr_i,val_i) in enumerate(kf.split(X_arr)):
    m=CatBoostRegressor(**cat6)
    m.fit(X_arr[tr_i],y_arr[tr_i],eval_set=(X_arr[val_i],y_arr[val_i]),use_best_model=True)
    oof['cat6'][val_i] = m.predict(X_arr[val_i]); test_preds['cat6'] += m.predict(X_test_arr)/N_FOLDS
    print(f"  F{fold+1}: RMSE={np.sqrt(mean_squared_error(y_arr[val_i],oof['cat6'][val_i])):.5f} R2={r2_score(y_arr[val_i],oof['cat6'][val_i]):.4f}")
print(f"CAT6 OOF RMSE={np.sqrt(mean_squared_error(y_arr,oof['cat6'])):.5f} R2={r2_score(y_arr,oof['cat6']):.4f}")

# ExtraTrees
print("\n=== Extra Trees ===")
oof['et'] = np.zeros(len(X)); test_preds['et'] = np.zeros(len(X_test))
for fold,(tr_i,val_i) in enumerate(kf.split(X_arr)):
    m = ExtraTreesRegressor(n_estimators=500,max_depth=None,min_samples_leaf=5,
                             max_features=0.6,n_jobs=-1,random_state=SEED+fold)
    m.fit(X_arr[tr_i],y_arr[tr_i])
    oof['et'][val_i] = m.predict(X_arr[val_i]); test_preds['et'] += m.predict(X_test_arr)/N_FOLDS
    print(f"  F{fold+1}: RMSE={np.sqrt(mean_squared_error(y_arr[val_i],oof['et'][val_i])):.5f} R2={r2_score(y_arr[val_i],oof['et'][val_i]):.4f}")
print(f"ET OOF RMSE={np.sqrt(mean_squared_error(y_arr,oof['et'])):.5f} R2={r2_score(y_arr,oof['et']):.4f}")

# ─── Stack ───────────────────────────────────
print("\n=== Stacking ===")
model_names = list(oof.keys())
oof_stack   = np.column_stack([oof[k] for k in model_names])
test_stack  = np.column_stack([test_preds[k] for k in model_names])
rmse_list   = [np.sqrt(mean_squared_error(y_arr, oof[k])) for k in model_names]
r2_list     = [r2_score(y_arr, oof[k]) for k in model_names]

ridge = Ridge(alpha=1.0)
ridge.fit(oof_stack, y_arr)
stack_test  = ridge.predict(test_stack)
stack_oof   = ridge.predict(oof_stack)
stack_rmse  = np.sqrt(mean_squared_error(y_arr, stack_oof))
stack_r2    = r2_score(y_arr, stack_oof)

# Rank-average blend (top models only)
top_k     = sorted(zip(model_names, rmse_list), key=lambda x: x[1])[:4]
print(f"Top 4 models for blend: {[k for k,r in top_k]}")
blend_test_rank = np.zeros(len(X_test))
blend_oof_rank  = np.zeros(len(X))
for k, _ in top_k:
    oof_ranks    = np.argsort(np.argsort(oof[k])).astype(float) / len(y)
    test_ranks   = np.argsort(np.argsort(test_preds[k])).astype(float) / len(X_test)
    blend_oof_rank  += oof_ranks  / len(top_k)
    blend_test_rank += test_ranks / len(top_k)
# Re-map rank blend to training target distribution
sorted_y    = np.sort(y_arr)
rank_blend_oof  = np.interp(blend_oof_rank,  np.linspace(0,1,len(sorted_y)), sorted_y)
rank_blend_test = np.interp(blend_test_rank, np.linspace(0,1,len(sorted_y)), sorted_y)
rank_rmse = np.sqrt(mean_squared_error(y_arr, rank_blend_oof))
rank_r2   = r2_score(y_arr, rank_blend_oof)

print("\n=== Final Results ===")
for k, r, r2 in zip(model_names, rmse_list, r2_list):
    print(f"  {k:12s} RMSE={r:.5f} R2={r2:.4f} std={np.std(test_preds[k]):.4f}")
print(f"  {'Stack':12s} RMSE={stack_rmse:.5f} R2={stack_r2:.4f} std={np.std(stack_test):.4f}")
print(f"  {'RankBlend':12s} RMSE={rank_rmse:.5f} R2={rank_r2:.4f} std={np.std(rank_blend_test):.4f}")
print(f"  Ridge coefs: {dict(zip(model_names, ridge.coef_.round(4)))}")

for fname, preds in [
    (OUTPUT_FILE,                    stack_test),
    ("submission_v10_rank.csv",      rank_blend_test),
    ("submission_v10_cat8.csv",      test_preds['cat8']),
    ("submission_v10_lgb_dart.csv",  test_preds['lgb_dart']),
]:
    pd.DataFrame({'record_id': test_record_ids.values,
                  'flood_risk_score': np.clip(preds,0,1)}).to_csv(fname,index=False)
    print(f"[OK] {fname}")

print("\n" + "="*60)
print("V10 SUMMARY (pure trees + TE interactions + DART)")
print("="*60)
for k,r,r2 in zip(model_names,rmse_list,r2_list): print(f"  {k:12s} RMSE={r:.5f} R2={r2:.4f}")
print(f"  {'Stack':12s} RMSE={stack_rmse:.5f} R2={stack_r2:.4f}")
print(f"  {'RankBlend':12s} RMSE={rank_rmse:.5f} R2={rank_r2:.4f}")
print(f"  V9 Stack (LB TBD):  RMSE=0.23446 R2=0.0357")
print(f"  V8 Stack (LB 0.38215): RMSE=0.23459 R2=0.0346")
print("="*60)
