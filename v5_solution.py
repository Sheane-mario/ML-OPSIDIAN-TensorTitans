"""
V5 Solution: CRITICAL FIX - Place-name target encoding is the strongest signal
Key insight: All 792 test place_names overlap with training.
Predicting place mean alone → RMSE 0.23417 (better than our v2 ensemble 0.23497!)

V5 improvements:
  1. Place-name target encoding (fold-safe, smoothed Bayesian) - the most important feature
  2. Include all "dropped" features: is_good_to_live, inundation_area_sqm, reason_not_good_to_live
  3. Place-level statistics: mean, std, p25, p75 of flood_risk by place
  4. generation_date features: month, year, day_of_year, season
  5. reason_not_good_to_live multi-label binary flags
  6. All engineered features from v4
  7. CatBoost primary (consistently best) + LGB + XGB ensemble
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

SEED        = 42
N_FOLDS     = 10
DATA_DIR    = "data"
OUTPUT_FILE = "submission_v5.csv"
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Train: {train.shape}, Test: {test.shape}")

TARGET = 'flood_risk_score'
test_record_ids = test['record_id'].copy()

# ─────────────────────────────────────────────
# 2. Parse generation_date features
# ─────────────────────────────────────────────
print("Parsing date features...")
for df in [train, test]:
    df['generation_date'] = pd.to_datetime(df['generation_date'])
    df['gen_month']      = df['generation_date'].dt.month
    df['gen_year']       = df['generation_date'].dt.year
    df['gen_day_of_year']= df['generation_date'].dt.dayofyear
    df['gen_quarter']    = df['generation_date'].dt.quarter
    # Sri Lanka monsoon seasons
    df['is_northeast_monsoon'] = df['gen_month'].isin([12, 1, 2]).astype(int)
    df['is_southwest_monsoon'] = df['gen_month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['gen_month_sin']  = np.sin(2 * np.pi * df['gen_month'] / 12)
    df['gen_month_cos']  = np.cos(2 * np.pi * df['gen_month'] / 12)

# ─────────────────────────────────────────────
# 3. Reason flags from reason_not_good_to_live
# ─────────────────────────────────────────────
print("Parsing reason flags...")
for df in [train, test]:
    reason = df['reason_not_good_to_live'].fillna('Other')
    df['reason_flood_flag'] = reason.str.contains('flood', case=False).astype(int)
    df['reason_infra_flag'] = reason.str.contains('infrastructure', case=False).astype(int)
    df['reason_road_flag']  = reason.str.contains('road', case=False).astype(int)
    df['reason_other_flag'] = (reason == 'Other').astype(int)
    df['is_good_to_live_bin'] = (df['is_good_to_live'] == 'Yes').astype(int)

# ─────────────────────────────────────────────
# 4. inundation_area_sqm transforms
# ─────────────────────────────────────────────
for df in [train, test]:
    df['log_inundation']    = np.log1p(df['inundation_area_sqm'])
    df['sqrt_inundation']   = np.sqrt(df['inundation_area_sqm'])
    df['inundation_per_pop']= df['inundation_area_sqm'] / (df['population_density_per_km2'] + 1)

# ─────────────────────────────────────────────
# 5. Standard Feature Engineering
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    eps = 1e-6

    df['rainfall_x_flood_count']  = df['rainfall_7d_mm'] * df['historical_flood_count']
    df['monthly_x_flood_count']   = df['monthly_rainfall_mm'] * df['historical_flood_count']
    df['rain_ratio_7d_monthly']   = df['rainfall_7d_mm'] / (df['monthly_rainfall_mm'] + eps)
    df['rainfall_cumulative']     = df['rainfall_7d_mm'] + df['monthly_rainfall_mm']

    df['river_dist_clip']         = df['distance_to_river_m'].clip(lower=0)
    df['river_rainfall_risk']     = df['rainfall_7d_mm'] / (df['river_dist_clip'] + 1)
    df['river_monthly_risk']      = df['monthly_rainfall_mm'] / (df['river_dist_clip'] + 1)

    df['elev_clip']               = df['elevation_m'].clip(lower=0)
    df['elev_rainfall_ratio']     = df['rainfall_7d_mm'] / (df['elev_clip'] + 1)
    df['low_elevation_flag']      = (df['elevation_m'] < 30).astype(int)
    df['very_low_elevation_flag'] = (df['elevation_m'] < 10).astype(int)

    df['infra_socio_product']     = df['infrastructure_score'] * df['socioeconomic_status_index']
    df['water_veg_balance']       = df['ndwi'] - df['ndvi']
    df['ndvi_ndwi_product']       = df['ndvi'] * df['ndwi']

    df['drainage_rain_ratio']     = df['drainage_index'] / (df['rainfall_7d_mm'] + 1)
    df['drainage_x_rain']         = df['drainage_index'] * df['rainfall_7d_mm']
    df['poor_drainage_flag']      = (df['drainage_index'] < 0.35).astype(int)
    df['poor_drainage_rain']      = df['poor_drainage_flag'] * df['rainfall_7d_mm']

    df['urban_flood_amplifier']   = df['built_up_percent'] * df['rainfall_7d_mm'] / 100
    df['hospital_evac_sum']       = df['nearest_hospital_km'] + df['nearest_evac_km']
    df['max_distance_to_help']    = df[['nearest_hospital_km', 'nearest_evac_km']].max(axis=1)
    df['pop_density_x_rain']      = df['population_density_per_km2'] * df['rainfall_7d_mm']
    df['pop_density_x_flood']     = df['population_density_per_km2'] * df['historical_flood_count']

    df['extreme_x_rain']          = df['extreme_weather_index'] * df['rainfall_7d_mm']
    df['extreme_x_flood']         = df['extreme_weather_index'] * df['historical_flood_count']
    df['extreme_x_monthly']       = df['extreme_weather_index'] * df['monthly_rainfall_mm']

    df['seasonal_rain']           = df['seasonal_index'] * df['rainfall_7d_mm']
    df['seasonal_x_extreme']      = df['seasonal_index'] * df['extreme_weather_index']
    df['terrain_rain_interaction']= df['terrain_roughness_index'] * df['rainfall_7d_mm']

    # Cross with log-transformed cols (pre-computed in dataset)
    if 'rainfall_7d_mm_log1p' in df.columns:
        df['log_rain_x_flood']    = df['rainfall_7d_mm_log1p'] * df['historical_flood_count']
        df['log_rain_x_extreme']  = df['rainfall_7d_mm_log1p'] * df['extreme_weather_index']
    if 'distance_to_river_m_log1p' in df.columns:
        df['log_river_x_rain']    = df['distance_to_river_m_log1p'] * df['rainfall_7d_mm']

    df['composite_vulnerability'] = (
        df['rainfall_7d_mm']         * 0.25 +
        df['historical_flood_count'] * 15.0 +
        df['extreme_weather_index']  * 50.0 +
        (1 - df['drainage_index'])   * 30.0 +
        df['built_up_percent']       * 0.10
    )

    # Inundation × rainfall interaction (new!)
    df['inundation_x_rain']       = df['inundation_area_sqm'] * df['rainfall_7d_mm']
    df['log_inundation_x_rain']   = df['log_inundation'] * df['rainfall_7d_mm']

    return df

print("Engineering features...")
train = engineer_features(train)
test  = engineer_features(test)

# ─────────────────────────────────────────────
# 6. Place-Level Statistics (from training only - no leakage)
# ─────────────────────────────────────────────
print("Computing place-level statistics...")
place_stats = train.groupby('place_name')[TARGET].agg(
    place_mean_risk   = 'mean',
    place_median_risk = 'median',
    place_std_risk    = 'std',
    place_min_risk    = 'min',
    place_max_risk    = 'max',
    place_count       = 'count',
    place_p25_risk    = lambda x: x.quantile(0.25),
    place_p75_risk    = lambda x: x.quantile(0.75),
    place_p10_risk    = lambda x: x.quantile(0.10),
    place_p90_risk    = lambda x: x.quantile(0.90),
).reset_index()
place_stats['place_range_risk'] = place_stats['place_max_risk'] - place_stats['place_min_risk']
place_stats['place_iqr_risk']   = place_stats['place_p75_risk'] - place_stats['place_p25_risk']
place_stats['place_std_risk']   = place_stats['place_std_risk'].fillna(0)

train = train.merge(place_stats, on='place_name', how='left')
test  = test.merge(place_stats,  on='place_name', how='left')

print(f"Place-mean correlation with target: {train['place_mean_risk'].corr(train[TARGET]):.4f}")
place_rmse = np.sqrt(((train[TARGET] - train['place_mean_risk'])**2).mean())
print(f"Place-mean alone RMSE: {place_rmse:.5f}")

# ─────────────────────────────────────────────
# 7. District-Level Statistics
# ─────────────────────────────────────────────
dist_stats = train.groupby('district')[TARGET].agg(
    dist_mean_risk  = 'mean',
    dist_median_risk= 'median',
    dist_std_risk   = 'std',
).reset_index().fillna(0)

train = train.merge(dist_stats, on='district', how='left')
test  = test.merge(dist_stats,  on='district', how='left')

# ─────────────────────────────────────────────
# 8. Label Encode Categoricals
# ─────────────────────────────────────────────
print("Encoding categoricals...")
CAT_COLS = [
    'district', 'landcover', 'soil_type', 'water_supply',
    'electricity', 'road_quality', 'urban_rural',
    'water_presence_flag', 'flood_occurrence_current_event',
    'is_good_to_live', 'reason_not_good_to_live', 'place_name'
]

DROP_COLS = ['record_id', 'generation_date', 'is_synthetic', TARGET]

all_data = pd.concat([train, test], axis=0, ignore_index=True)
for col in CAT_COLS:
    if col in all_data.columns:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))

n_train   = len(train)
train_enc = all_data.iloc[:n_train].copy()
test_enc  = all_data.iloc[n_train:].copy()
train_enc[TARGET] = train[TARGET].values

# ─────────────────────────────────────────────
# 9. Build Feature Set
# ─────────────────────────────────────────────
EXCLUDE      = set(DROP_COLS + [TARGET, 'record_id'])
feature_cols = [c for c in train_enc.columns if c not in EXCLUDE]
print(f"Total features before TE: {len(feature_cols)}")

X      = train_enc[feature_cols].copy()
y      = train_enc[TARGET].copy()
X_test = test_enc[feature_cols].copy()

medians = X.median(numeric_only=True)
X       = X.fillna(medians)
X_test  = X_test.fillna(medians)

# ─────────────────────────────────────────────
# 10. Fold-Safe Target Encoding: place_name AND district
# ─────────────────────────────────────────────
print("Computing fold-safe target encodings...")
kf_te = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
SMOOTH_PLACE    = 5   # low smoothing - many rows per place
SMOOTH_DISTRICT = 20
global_mean = y.mean()

def bayesian_te(X_tr_col, y_tr, X_val_col, X_test_col, smooth, global_mean):
    """Fold-safe Bayesian target encoding."""
    stats = pd.DataFrame({'k': X_tr_col.values, 'y': y_tr.values})
    stats = stats.groupby('k')['y'].agg(['mean', 'count'])
    stats['enc'] = (stats['mean'] * stats['count'] + global_mean * smooth) / (stats['count'] + smooth)
    val_enc  = X_val_col.map(stats['enc']).fillna(global_mean).values
    test_enc = X_test_col.map(stats['enc']).fillna(global_mean).values
    return val_enc, test_enc

# place_name target encoding
place_te_train = np.zeros(len(X))
place_te_test  = np.zeros(len(X_test))
for tr_idx, val_idx in kf_te.split(X.values):
    val, _ = bayesian_te(X.iloc[tr_idx]['place_name'], y.iloc[tr_idx],
                         X.iloc[val_idx]['place_name'], X_test['place_name'],
                         SMOOTH_PLACE, global_mean)
    place_te_train[val_idx] = val

# Full-data for test
_, place_te_test = bayesian_te(X['place_name'], y, X['place_name'], X_test['place_name'],
                               SMOOTH_PLACE, global_mean)

# district target encoding
district_te_train = np.zeros(len(X))
district_te_test  = np.zeros(len(X_test))
for tr_idx, val_idx in kf_te.split(X.values):
    val, _ = bayesian_te(X.iloc[tr_idx]['district'], y.iloc[tr_idx],
                         X.iloc[val_idx]['district'], X_test['district'],
                         SMOOTH_DISTRICT, global_mean)
    district_te_train[val_idx] = val
_, district_te_test = bayesian_te(X['district'], y, X['district'], X_test['district'],
                                  SMOOTH_DISTRICT, global_mean)

# Append TE columns
X['place_target_enc']    = place_te_train
X['district_target_enc'] = district_te_train
X_test['place_target_enc']    = place_te_test
X_test['district_target_enc'] = district_te_test
feature_cols = list(X.columns)

X_arr      = X.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)
y_arr      = y.values.astype(np.float32)

print(f"Final features: {len(feature_cols)}")
print(f"X_arr NaN: {np.isnan(X_arr).any()}")

# ─────────────────────────────────────────────
# 11. Train Ensemble (10-fold)
# ─────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_et  = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_et  = np.zeros(len(X_test))

# ─── LightGBM ─────────────────────────────────
print("\n=== LightGBM (10-fold) ===")
lgb_params = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.05, 'num_leaves': 63,
    'min_child_samples': 20, 'feature_fraction': 0.7,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'n_jobs': -1, 'verbose': -1, 'seed': SEED,
}
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=feature_cols)
    dval   = lgb.Dataset(Xvl, label=yvl, reference=dtrain)
    model  = lgb.train(lgb_params, dtrain, num_boost_round=3000, valid_sets=[dval],
                       callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(9999)])
    oof_lgb[val_idx] = model.predict(Xvl)
    test_lgb        += model.predict(X_test_arr) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(yvl, oof_lgb[val_idx]))
    print(f"  Fold {fold+1:2d}: RMSE={rmse:.5f} | iter={model.best_iteration}")

lgb_rmse = np.sqrt(mean_squared_error(y_arr, oof_lgb))
print(f"LGB OOF RMSE: {lgb_rmse:.5f}")

# ─── XGBoost ──────────────────────────────────
print("\n=== XGBoost (10-fold) ===")
xgb_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'learning_rate': 0.05, 'max_depth': 7,
    'min_child_weight': 10, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'gamma': 0.1,
    'n_jobs': -1, 'seed': SEED, 'tree_method': 'hist', 'verbosity': 0,
}
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_cols)
    dval   = xgb.DMatrix(Xvl, label=yvl, feature_names=feature_cols)
    dtest  = xgb.DMatrix(X_test_arr, feature_names=feature_cols)
    model  = xgb.train(xgb_params, dtrain, num_boost_round=3000,
                       evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=9999)
    oof_xgb[val_idx] = model.predict(dval)
    test_xgb        += model.predict(dtest) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(yvl, oof_xgb[val_idx]))
    print(f"  Fold {fold+1:2d}: RMSE={rmse:.5f} | iter={model.best_iteration}")

xgb_rmse = np.sqrt(mean_squared_error(y_arr, oof_xgb))
print(f"XGB OOF RMSE: {xgb_rmse:.5f}")

# ─── CatBoost ─────────────────────────────────
print("\n=== CatBoost (10-fold) ===")
cat_params = dict(
    iterations=3000, learning_rate=0.05, depth=8,
    l2_leaf_reg=5, min_data_in_leaf=20,
    subsample=0.8, colsample_bylevel=0.7,
    random_seed=SEED, task_type='CPU', verbose=0,
    eval_metric='RMSE', early_stopping_rounds=100,
)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    model = CatBoostRegressor(**cat_params)
    model.fit(Xtr, ytr, eval_set=(Xvl, yvl), use_best_model=True)
    oof_cat[val_idx] = model.predict(Xvl)
    test_cat        += model.predict(X_test_arr) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(yvl, oof_cat[val_idx]))
    print(f"  Fold {fold+1:2d}: RMSE={rmse:.5f} | iter={model.best_iteration_}")

cat_rmse = np.sqrt(mean_squared_error(y_arr, oof_cat))
print(f"CAT OOF RMSE: {cat_rmse:.5f}")

# ─── Extra Trees ──────────────────────────────
print("\n=== Extra Trees (10-fold) ===")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    et = ExtraTreesRegressor(n_estimators=400, max_depth=None, min_samples_leaf=5,
                              max_features=0.6, n_jobs=-1, random_state=SEED+fold)
    et.fit(Xtr, ytr)
    oof_et[val_idx] = et.predict(Xvl)
    test_et        += et.predict(X_test_arr) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(yvl, oof_et[val_idx]))
    print(f"  Fold {fold+1:2d}: RMSE={rmse:.5f}")

et_rmse = np.sqrt(mean_squared_error(y_arr, oof_et))
print(f"ET OOF RMSE: {et_rmse:.5f}")

# ─────────────────────────────────────────────
# 12. Stacking & Blending
# ─────────────────────────────────────────────
print("\n=== Stacking & Blending ===")
oof_all  = [oof_lgb, oof_xgb, oof_cat, oof_et]
test_all = [test_lgb, test_xgb, test_cat, test_et]
rmse_all = [lgb_rmse, xgb_rmse, cat_rmse, et_rmse]
lbls     = ['LGB', 'XGB', 'CAT', 'ET']

oof_stack  = np.column_stack(oof_all)
test_stack = np.column_stack(test_all)

# Ridge meta-learner
ridge = Ridge(alpha=1.0)
ridge.fit(oof_stack, y_arr)
stack_oof    = ridge.predict(oof_stack)
stack_test   = ridge.predict(test_stack)
stack_rmse   = np.sqrt(mean_squared_error(y_arr, stack_oof))

# LGB meta-learner  
oof_lgb_meta  = np.zeros(len(X))
test_lgb_meta = np.zeros(len(X_test))
kf_meta = KFold(n_splits=5, shuffle=True, random_state=SEED+99)
meta_params = {'objective': 'regression', 'metric': 'rmse', 'num_leaves': 8,
               'learning_rate': 0.05, 'n_jobs': -1, 'verbose': -1,
               'seed': SEED, 'reg_lambda': 2.0, 'min_child_samples': 10}
for tr_idx, val_idx in kf_meta.split(oof_stack):
    dt = lgb.Dataset(oof_stack[tr_idx], label=y_arr[tr_idx])
    dv = lgb.Dataset(oof_stack[val_idx], label=y_arr[val_idx], reference=dt)
    m  = lgb.train(meta_params, dt, 1000, valid_sets=[dv],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)])
    oof_lgb_meta[val_idx]  = m.predict(oof_stack[val_idx])
    test_lgb_meta         += m.predict(test_stack) / 5
lgb_meta_rmse = np.sqrt(mean_squared_error(y_arr, oof_lgb_meta))

# Weighted blend
inv_rmse   = np.array([1.0/r for r in rmse_all])
weights    = inv_rmse / inv_rmse.sum()
blend_oof  = sum(w*p for w,p in zip(weights, oof_all))
blend_test = sum(w*p for w,p in zip(weights, test_all))
blend_rmse = np.sqrt(mean_squared_error(y_arr, blend_oof))

candidates = {
    'Ridge-Stack':   (stack_rmse,     stack_test),
    'LGB-Meta':      (lgb_meta_rmse,  test_lgb_meta),
    'Blend':         (blend_rmse,     blend_test),
    'CAT-only':      (cat_rmse,       test_cat),
    'LGB-only':      (lgb_rmse,       test_lgb),
    'Mix(R+B 0.5)':  (
        np.sqrt(mean_squared_error(y_arr, 0.5*stack_oof+0.5*blend_oof)),
        0.5*stack_test + 0.5*blend_test
    ),
}

print("\n=== Candidate OOF RMSEs ===")
best_name, best_rmse, best_preds = None, np.inf, None
for name, (rmse, preds) in candidates.items():
    std = np.std(preds)
    print(f"  {name:20s}: RMSE={rmse:.5f}, pred_std={std:.4f}, range=[{preds.min():.3f},{preds.max():.3f}]")
    if rmse < best_rmse:
        best_name, best_rmse, best_preds = name, rmse, preds

print(f"\nBest: {best_name} (OOF RMSE={best_rmse:.5f})")
final_preds = np.clip(best_preds, 0.0, 1.0)

# ─────────────────────────────────────────────
# 13. Save
# ─────────────────────────────────────────────
pd.DataFrame({'record_id': test_record_ids.values, 'flood_risk_score': final_preds}
             ).to_csv(OUTPUT_FILE, index=False)

print(f"\n[OK] Saved: {OUTPUT_FILE}")
print(f"Pred stats: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}, "
      f"min={final_preds.min():.4f}, max={final_preds.max():.4f}")

print("\n" + "="*60)
print("FINAL SUMMARY (V5 - with place_name target encoding)")
print("="*60)
for lbl, r in zip(lbls, rmse_all):
    print(f"  {lbl:15s} OOF RMSE: {r:.5f}")
print(f"  {'Ridge-Stack':15s} OOF RMSE: {stack_rmse:.5f}")
print(f"  {'LGB-Meta':15s} OOF RMSE: {lgb_meta_rmse:.5f}")
print(f"  {'Blend':15s} OOF RMSE: {blend_rmse:.5f}")
print(f"\n  WINNER: {best_name} -> {best_rmse:.5f}")
print(f"  (v2 best was 0.23497)")
print("="*60)
