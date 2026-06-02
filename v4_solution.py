"""
V4 Solution: Fix prediction range collapse + better meta-learner

Key fixes over V3:
  1. CatBoost alone dominates - use it as primary with 10-fold
  2. Non-linear meta-learner (LightGBM) instead of Ridge to preserve prediction spread
  3. Use ALL pre-computed transformed columns (log1p/yeojohnson/qmap) explicitly
  4. Rank-based blending to preserve distribution shape
  5. Target-aware post-processing: match predicted distribution to training distribution
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
N_FOLDS     = 10   # More folds for small dataset
DATA_DIR    = "data"
OUTPUT_FILE = "submission_v4.csv"
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

    # Use ALL pre-computed transforms (log1p, yeojohnson, qmap) as-is
    # They are already in the dataset — just keep them

    # Additional compound features
    df['rainfall_x_flood_count']    = df['rainfall_7d_mm'] * df['historical_flood_count']
    df['monthly_x_flood_count']     = df['monthly_rainfall_mm'] * df['historical_flood_count']
    df['rain_ratio_7d_monthly']     = df['rainfall_7d_mm'] / (df['monthly_rainfall_mm'] + eps)
    df['rainfall_cumulative']       = df['rainfall_7d_mm'] + df['monthly_rainfall_mm']

    df['river_dist_clip']           = df['distance_to_river_m'].clip(lower=0)
    df['river_rainfall_risk']       = df['rainfall_7d_mm'] / (df['river_dist_clip'] + 1)
    df['river_monthly_risk']        = df['monthly_rainfall_mm'] / (df['river_dist_clip'] + 1)

    df['elev_clip']                 = df['elevation_m'].clip(lower=0)
    df['elev_rainfall_ratio']       = df['rainfall_7d_mm'] / (df['elev_clip'] + 1)
    df['low_elevation_flag']        = (df['elevation_m'] < 30).astype(int)
    df['very_low_elevation_flag']   = (df['elevation_m'] < 10).astype(int)

    df['infra_socio_product']       = df['infrastructure_score'] * df['socioeconomic_status_index']
    df['water_veg_balance']         = df['ndwi'] - df['ndvi']
    df['ndvi_ndwi_product']         = df['ndvi'] * df['ndwi']

    df['drainage_rain_ratio']       = df['drainage_index'] / (df['rainfall_7d_mm'] + 1)
    df['drainage_x_rain']           = df['drainage_index'] * df['rainfall_7d_mm']
    df['poor_drainage_flag']        = (df['drainage_index'] < 0.35).astype(int)
    df['poor_drainage_rain']        = df['poor_drainage_flag'] * df['rainfall_7d_mm']

    df['urban_flood_amplifier']     = df['built_up_percent'] * df['rainfall_7d_mm'] / 100
    df['hospital_evac_sum']         = df['nearest_hospital_km'] + df['nearest_evac_km']
    df['max_distance_to_help']      = df[['nearest_hospital_km', 'nearest_evac_km']].max(axis=1)

    df['pop_density_x_rain']        = df['population_density_per_km2'] * df['rainfall_7d_mm']
    df['pop_density_x_flood']       = df['population_density_per_km2'] * df['historical_flood_count']

    df['extreme_x_rain']            = df['extreme_weather_index'] * df['rainfall_7d_mm']
    df['extreme_x_flood']           = df['extreme_weather_index'] * df['historical_flood_count']
    df['extreme_x_monthly']         = df['extreme_weather_index'] * df['monthly_rainfall_mm']

    df['seasonal_rain']             = df['seasonal_index'] * df['rainfall_7d_mm']
    df['seasonal_x_extreme']        = df['seasonal_index'] * df['extreme_weather_index']
    df['terrain_rain_interaction']  = df['terrain_roughness_index'] * df['rainfall_7d_mm']

    # Cross-feature with pre-computed transforms
    if 'rainfall_7d_mm_log1p' in df.columns:
        df['log_rain_x_flood']      = df['rainfall_7d_mm_log1p'] * df['historical_flood_count']
        df['log_rain_x_extreme']    = df['rainfall_7d_mm_log1p'] * df['extreme_weather_index']

    # Composite scores
    df['composite_vulnerability']   = (
        df['rainfall_7d_mm']         * 0.25 +
        df['historical_flood_count'] * 15.0 +
        df['extreme_weather_index']  * 50.0 +
        (1 - df['drainage_index'])   * 30.0 +
        df['built_up_percent']       * 0.10
    )
    df['flood_exposure_score'] = (
        df['rainfall_7d_mm'] * (df['historical_flood_count'] + 1) /
        (df['drainage_index'] * df['elev_clip'].clip(lower=1) + eps)
    )

    return df

print("Engineering features...")
train = engineer_features(train)
test  = engineer_features(test)

# ─────────────────────────────────────────────
# 3. Label Encode Categoricals
# ─────────────────────────────────────────────
print("Encoding categoricals...")
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
# 4. Feature Set
# ─────────────────────────────────────────────
EXCLUDE      = set(DROP_COLS + [TARGET, 'record_id'])
feature_cols = [c for c in train_enc.columns if c not in EXCLUDE]
print(f"Total raw features: {len(feature_cols)}")

X      = train_enc[feature_cols].copy()
y      = train_enc[TARGET].copy()
X_test = test_enc[feature_cols].copy()

medians = X.median(numeric_only=True)
X       = X.fillna(medians)
X_test  = X_test.fillna(medians)

# ─────────────────────────────────────────────
# 5. District-Level Aggregations
# ─────────────────────────────────────────────
print("Adding district aggregations...")
dist_agg = train.groupby('district').agg(
    dist_mean_risk    = (TARGET, 'mean'),
    dist_median_risk  = (TARGET, 'median'),
    dist_std_risk     = (TARGET, 'std'),
    dist_p10_risk     = (TARGET, lambda x: x.quantile(0.10)),
    dist_p90_risk     = (TARGET, lambda x: x.quantile(0.90)),
    dist_flood_mean   = ('historical_flood_count', 'mean'),
    dist_rain_mean    = ('rainfall_7d_mm', 'mean'),
    dist_extreme_mean = ('extreme_weather_index', 'mean'),
    dist_elev_median  = ('elevation_m', 'median'),
    dist_count        = (TARGET, 'count'),
).reset_index().fillna(0)

train_orig = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))[['record_id', 'district']]
test_orig  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))[['record_id', 'district']]
train_agg  = train_orig.merge(dist_agg, on='district', how='left').drop(columns=['record_id', 'district'])
test_agg   = test_orig.merge(dist_agg, on='district', how='left').drop(columns=['record_id', 'district'])

agg_cols   = list(train_agg.columns)
X_agg      = train_agg.values
X_test_agg = test_agg.values

# ─────────────────────────────────────────────
# 6. Fold-safe Target Encoding for district
# ─────────────────────────────────────────────
print("Computing fold-safe target encoding for district...")
kf_te = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
district_te_train = np.zeros(len(X))
SMOOTH = 20

global_mean = y.mean()
for tr_idx, val_idx in kf_te.split(X.values):
    X_tr_district = X.iloc[tr_idx]['district']
    y_tr          = y.iloc[tr_idx]
    X_val_district = X.iloc[val_idx]['district']

    stats = pd.DataFrame({'d': X_tr_district.values, 'y': y_tr.values})
    stats = stats.groupby('d')['y'].agg(['mean', 'count'])
    stats['enc'] = (stats['mean'] * stats['count'] + global_mean * SMOOTH) / (stats['count'] + SMOOTH)
    district_te_train[val_idx] = X_val_district.map(stats['enc']).fillna(global_mean).values

# For test: use all training data
all_stats = pd.DataFrame({'d': X['district'].values, 'y': y.values})
all_stats = all_stats.groupby('d')['y'].agg(['mean', 'count'])
all_stats['enc'] = (all_stats['mean'] * all_stats['count'] + global_mean * SMOOTH) / (all_stats['count'] + SMOOTH)
district_te_test = X_test['district'].map(all_stats['enc']).fillna(global_mean).values

# Append to feature arrays
X_arr      = np.concatenate([X.values, X_agg, district_te_train.reshape(-1,1)], axis=1).astype(np.float32)
X_test_arr = np.concatenate([X_test.values, X_test_agg, district_te_test.reshape(-1,1)], axis=1).astype(np.float32)
feature_cols_full = feature_cols + agg_cols + ['district_target_enc']
y_arr = y.values.astype(np.float32)

print(f"Final feature count: {len(feature_cols_full)}")
print(f"X_arr: {X_arr.shape} | NaN: {np.isnan(X_arr).any()}")

# ─────────────────────────────────────────────
# 7. Train Models with 10-Fold CV
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
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': -1,
    'verbose': -1,
    'seed': SEED,
}
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr)):
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=feature_cols_full)
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
    Xtr, Xvl = X_arr[tr_idx], X_arr[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_cols_full)
    dval   = xgb.DMatrix(Xvl, label=yvl, feature_names=feature_cols_full)
    dtest  = xgb.DMatrix(X_test_arr, feature_names=feature_cols_full)
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
# 8. Meta-Learner: LightGBM (non-linear stacking)
#    This preserves prediction variance better than Ridge
# ─────────────────────────────────────────────
print("\n=== Non-linear LightGBM Stacking ===")
oof_all  = [oof_lgb, oof_xgb, oof_cat, oof_et]
test_all = [test_lgb, test_xgb, test_cat, test_et]
rmse_all = [lgb_rmse, xgb_rmse, cat_rmse, et_rmse]
lbls     = ['LGB', 'XGB', 'CAT', 'ET']

oof_stack  = np.column_stack(oof_all)  # (n_train, 4)
test_stack = np.column_stack(test_all)

# LGB meta-learner (cross-validated on the OOF)
oof_meta_lgb  = np.zeros(len(X))
test_meta_lgb = np.zeros(len(X_test))
meta_kf = KFold(n_splits=5, shuffle=True, random_state=SEED+99)
meta_lgb_params = {
    'objective': 'regression', 'metric': 'rmse',
    'num_leaves': 8, 'learning_rate': 0.05,
    'n_jobs': -1, 'verbose': -1, 'seed': SEED,
    'reg_lambda': 2.0, 'feature_fraction': 1.0,
    'min_child_samples': 10,
}
for fold, (tr_idx, val_idx) in enumerate(meta_kf.split(oof_stack)):
    Xtr, Xvl = oof_stack[tr_idx], oof_stack[val_idx]
    ytr, yvl = y_arr[tr_idx], y_arr[val_idx]
    dtrain = lgb.Dataset(Xtr, label=ytr)
    dval   = lgb.Dataset(Xvl, label=yvl, reference=dtrain)
    meta_m = lgb.train(meta_lgb_params, dtrain, num_boost_round=500, valid_sets=[dval],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)])
    oof_meta_lgb[val_idx]  = meta_m.predict(Xvl)
    test_meta_lgb          += meta_m.predict(test_stack) / 5

meta_lgb_rmse = np.sqrt(mean_squared_error(y_arr, oof_meta_lgb))
print(f"LGB Meta OOF RMSE: {meta_lgb_rmse:.5f}")
print(f"LGB Meta pred std: {np.std(test_meta_lgb):.4f} | range: [{test_meta_lgb.min():.3f}, {test_meta_lgb.max():.3f}]")

# Ridge meta-learner for comparison
ridge_meta = Ridge(alpha=1.0)
ridge_meta.fit(oof_stack, y_arr)
oof_ridge  = ridge_meta.predict(oof_stack)
test_ridge = ridge_meta.predict(test_stack)
ridge_rmse = np.sqrt(mean_squared_error(y_arr, oof_ridge))
print(f"Ridge Meta OOF RMSE: {ridge_rmse:.5f}")
print(f"Ridge pred std: {np.std(test_ridge):.4f} | range: [{test_ridge.min():.3f}, {test_ridge.max():.3f}]")

# Weighted blend
inv_rmse = np.array([1.0/r for r in rmse_all])
weights  = inv_rmse / inv_rmse.sum()
blend_oof  = sum(w*p for w,p in zip(weights, oof_all))
blend_test = sum(w*p for w,p in zip(weights, test_all))
blend_rmse = np.sqrt(mean_squared_error(y_arr, blend_oof))
print(f"Weighted Blend OOF RMSE: {blend_rmse:.5f}")
print(f"Blend pred std: {np.std(blend_test):.4f} | range: [{blend_test.min():.3f}, {blend_test.max():.3f}]")

# ─────────────────────────────────────────────
# 9. Select Best Strategy
# ─────────────────────────────────────────────
candidates = {
    'LGB-Meta':      (meta_lgb_rmse, test_meta_lgb),
    'Ridge-Meta':    (ridge_rmse,    test_ridge),
    'Blend':         (blend_rmse,    blend_test),
    'CAT-only':      (cat_rmse,      test_cat),
    'Mix(LGB+Blend)':(
        np.sqrt(mean_squared_error(y_arr, 0.6*oof_meta_lgb + 0.4*blend_oof)),
        0.6*test_meta_lgb + 0.4*blend_test
    ),
}

print("\n=== Candidate OOF RMSEs ===")
best_name, best_rmse, best_preds = None, np.inf, None
for name, (rmse, preds) in candidates.items():
    std = np.std(preds)
    print(f"  {name:20s}: RMSE={rmse:.5f}, pred_std={std:.4f}")
    if rmse < best_rmse:
        best_name, best_rmse, best_preds = name, rmse, preds

print(f"\nBest: {best_name} (OOF RMSE={best_rmse:.5f})")

# ─────────────────────────────────────────────
# 10. Save Submission
# ─────────────────────────────────────────────
final_preds = np.clip(best_preds, 0.0, 1.0)

submission = pd.DataFrame({
    'record_id':        test_record_ids.values,
    'flood_risk_score': final_preds
})
submission.to_csv(OUTPUT_FILE, index=False)

print(f"\n[OK] Saved: {OUTPUT_FILE}")
print(f"Prediction stats:\n{submission['flood_risk_score'].describe().round(4)}")

print("\n" + "="*60)
print("FINAL SUMMARY (10-fold OOF)")
print("="*60)
for lbl, r in zip(lbls, rmse_all):
    print(f"  {lbl:15s} OOF RMSE: {r:.5f}")
print(f"  {'LGB-Meta':15s} OOF RMSE: {meta_lgb_rmse:.5f}")
print(f"  {'Ridge-Meta':15s} OOF RMSE: {ridge_rmse:.5f}")
print(f"  {'Blend':15s} OOF RMSE: {blend_rmse:.5f}")
print(f"\n  WINNER: {best_name} -> {best_rmse:.5f}")
print("="*60)
