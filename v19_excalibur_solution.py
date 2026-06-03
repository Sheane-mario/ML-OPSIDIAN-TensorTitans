"""
V19 Excalibur Ensemble

The multi-day, extreme scale training script.
Key Innovations:
1. Logit Target Transformation: Eliminates catastrophic outliers structurally.
2. Pseudo-Labeling: Uses V18's highly accurate test predictions as training data.
3. 60-Fold Cross-Validation: (20 folds x 3 repeats) for absolute maximum stability.
"""

import numpy as np
import pandas as pd
import os
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

import lightgbm as lgb
from catboost import CatBoostRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Skipping XGBoost models.")

SEED        = 42
N_FOLDS     = 20
N_REPEATS   = 3
DATA_DIR    = "data"
CHECKPOINT_DIR = "excalibur_checkpoints"
SUB_DIR     = "submissions"
PSEUDO_FILE = os.path.join(SUB_DIR, "submission_v18_titan.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
np.random.seed(SEED)

print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
pseudo = pd.read_csv(PSEUDO_FILE)
print(f"Train: {train.shape}, Test: {test.shape}")

# Inject Pseudo Labels into Test Set
test['flood_risk_score'] = pseudo['flood_risk_score'].values

# Track what is train and what is test
N_TRAIN = len(train)
TARGET = 'flood_risk_score'
test_record_ids = test['record_id'].copy()

# Concatenate for global feature engineering
all_data = pd.concat([train, test], axis=0, ignore_index=True)

# ─── Target Transformation ─────────────────────────────
EPS = 1e-6
def to_logit(y):
    y_clip = np.clip(y, EPS, 1 - EPS)
    return np.log(y_clip / (1 - y_clip))

def to_prob(x):
    # Safe sigmoid to avoid overflow
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

# We transform the target for training
all_data['logit_target'] = to_logit(all_data[TARGET].values)

# ─── Date / Meta Features ─────────────────────
all_data['gen_date']        = pd.to_datetime(all_data['generation_date'])
all_data['gen_month']       = all_data['gen_date'].dt.month
all_data['gen_year']        = all_data['gen_date'].dt.year
all_data['gen_day_of_year'] = all_data['gen_date'].dt.dayofyear
all_data['gen_quarter']     = all_data['gen_date'].dt.quarter
all_data['is_ne_monsoon']   = all_data['gen_month'].isin([12, 1, 2]).astype(int)
all_data['is_sw_monsoon']   = all_data['gen_month'].isin([5, 6, 7, 8, 9]).astype(int)
all_data['gen_month_sin']   = np.sin(2 * np.pi * all_data['gen_month'] / 12)
all_data['gen_month_cos']   = np.cos(2 * np.pi * all_data['gen_month'] / 12)
reason = all_data['reason_not_good_to_live'].fillna('Other')
all_data['reason_flood_flag'] = reason.str.contains('flood', case=False).astype(int)
all_data['reason_infra_flag'] = reason.str.contains('infrastructure', case=False).astype(int)
all_data['reason_road_flag']  = reason.str.contains('road', case=False).astype(int)
all_data['reason_other_flag'] = (reason == 'Other').astype(int)
all_data['is_good_binary']    = (all_data['is_good_to_live'] == 'Yes').astype(int)
all_data['log_inundation']    = np.log1p(all_data['inundation_area_sqm'])
all_data['sqrt_inundation']   = np.sqrt(all_data['inundation_area_sqm'])
all_data['inundation_per_pop']= all_data['inundation_area_sqm'] / (all_data['population_density_per_km2'] + 1)
all_data['record_id_num']     = all_data['record_id'].str.replace('F', '', regex=False).astype(int)

print("Engineering features...")
eps = 1e-6
all_data['rainfall_x_flood']       = all_data['rainfall_7d_mm'] * all_data['historical_flood_count']
all_data['monthly_x_flood']        = all_data['monthly_rainfall_mm'] * all_data['historical_flood_count']
all_data['rain_ratio']              = all_data['rainfall_7d_mm'] / (all_data['monthly_rainfall_mm'] + eps)
all_data['rain_cum']                = all_data['rainfall_7d_mm'] + all_data['monthly_rainfall_mm']
all_data['river_clip']              = all_data['distance_to_river_m'].clip(lower=0)
all_data['river_rain_risk']         = all_data['rainfall_7d_mm'] / (all_data['river_clip'] + 1)
all_data['river_monthly_risk']      = all_data['monthly_rainfall_mm'] / (all_data['river_clip'] + 1)
all_data['elev_clip']               = all_data['elevation_m'].clip(lower=0)
all_data['elev_rain_ratio']         = all_data['rainfall_7d_mm'] / (all_data['elev_clip'] + 1)
all_data['low_elev_flag']           = (all_data['elevation_m'] < 30).astype(int)
all_data['infra_socio']             = all_data['infrastructure_score'] * all_data['socioeconomic_status_index']
all_data['water_veg_balance']       = all_data['ndwi'] - all_data['ndvi']
all_data['ndvi_ndwi_product']       = all_data['ndvi'] * all_data['ndwi']
all_data['ndwi_sq']                 = all_data['ndwi'] ** 2
all_data['drainage_x_rain']         = all_data['drainage_index'] * all_data['rainfall_7d_mm']
all_data['bad_drainage_rain']       = (all_data['drainage_index'] < 0.35).astype(int) * all_data['rainfall_7d_mm']
all_data['urban_runoff']            = all_data['built_up_percent'] * all_data['rainfall_7d_mm'] / 100
all_data['evac_hosp_sum']           = all_data['nearest_hospital_km'] + all_data['nearest_evac_km']
all_data['max_dist_help']           = all_data[['nearest_hospital_km', 'nearest_evac_km']].max(axis=1)
all_data['pop_x_rain']              = all_data['population_density_per_km2'] * all_data['rainfall_7d_mm']
all_data['pop_x_flood']             = all_data['population_density_per_km2'] * all_data['historical_flood_count']
all_data['extreme_x_rain']          = all_data['extreme_weather_index'] * all_data['rainfall_7d_mm']
all_data['extreme_x_flood']         = all_data['extreme_weather_index'] * all_data['historical_flood_count']
all_data['extreme_x_monthly']       = all_data['extreme_weather_index'] * all_data['monthly_rainfall_mm']
all_data['seasonal_rain']           = all_data['seasonal_index'] * all_data['rainfall_7d_mm']
all_data['seasonal_extreme']        = all_data['seasonal_index'] * all_data['extreme_weather_index']
all_data['terrain_rain']            = all_data['terrain_roughness_index'] * all_data['rainfall_7d_mm']
all_data['log_rain_x_flood']        = np.log1p(all_data['rainfall_7d_mm']) * all_data['historical_flood_count']
all_data['log_rain_x_extreme']      = np.log1p(all_data['rainfall_7d_mm']) * all_data['extreme_weather_index']
all_data['log_river_x_rain']        = np.log1p(all_data['distance_to_river_m']) * all_data['rainfall_7d_mm']
all_data['inundation_x_rain']       = all_data['inundation_area_sqm'] * all_data['rainfall_7d_mm']
all_data['composite_vuln']          = (
    all_data['rainfall_7d_mm'] * 0.3 + all_data['historical_flood_count'] * 15.0 +
    all_data['extreme_weather_index'] * 50.0 + (1-all_data['drainage_index']) * 30.0 +
    all_data['built_up_percent'] * 0.10)

CAT_COLS = ['district','landcover','soil_type','water_supply','electricity',
            'road_quality','urban_rural','water_presence_flag',
            'flood_occurrence_current_event','is_good_to_live',
            'reason_not_good_to_live','place_name']
DROP_COLS = ['record_id','gen_date','generation_date','is_synthetic',TARGET,'logit_target']

for col in CAT_COLS:
    if col in all_data.columns:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))

EXCLUDE = set(DROP_COLS)
feature_cols = [c for c in all_data.columns if c not in EXCLUDE]
all_data[feature_cols] = all_data[feature_cols].fillna(all_data[feature_cols].median())

# Separate for Target Encoding
X      = all_data.iloc[:N_TRAIN][feature_cols].copy()
y      = all_data.iloc[:N_TRAIN]['logit_target'].copy() # Encode using LOGIT target!
X_test = all_data.iloc[N_TRAIN:][feature_cols].copy()
y_pseudo = all_data.iloc[N_TRAIN:]['logit_target'].copy()

# Feature Caching logic
TE_CACHE_FILE = os.path.join(CHECKPOINT_DIR, "features_cached.npz")
if os.path.exists(TE_CACHE_FILE):
    print("Loading cached 60-fold features...")
    data = np.load(TE_CACHE_FILE)
    X_arr = data['X_arr']
    X_test_arr = data['X_test_arr']
    y_logit_train = data['y_logit_train']
    y_logit_pseudo = data['y_logit_pseudo']
    y_true_train = data['y_true_train']
else:
    # Target Encoding using LOGIT Target!
    def fold_safe_te(X_df, y_s, X_te_df, col, smooth, gm, seed=SEED):
        tr_te = np.zeros(len(X_df))
        # Use single 20-fold for encoding to avoid infinite loops, model training handles 60
        kf = RepeatedKFold(n_splits=20, n_repeats=1, random_state=seed)
        for tr_idx, val_idx in kf.split(X_df):
            s = pd.DataFrame({'k': X_df.iloc[tr_idx][col].values, 'y': y_s.iloc[tr_idx].values}).groupby('k')['y'].agg(['mean','count'])
            s['enc'] = (s['mean']*s['count'] + gm*smooth) / (s['count']+smooth)
            tr_te[val_idx] = X_df.iloc[val_idx][col].map(s['enc']).fillna(gm).values
        s_all = pd.DataFrame({'k': X_df[col].values, 'y': y_s.values}).groupby('k')['y'].agg(['mean','count'])
        s_all['enc'] = (s_all['mean']*s_all['count'] + gm*smooth) / (s_all['count']+smooth)
        return tr_te, X_te_df[col].map(s_all['enc']).fillna(gm).values

    gm = y.mean()
    print(f"Computing fold-safe TEs + interactions...")
    te_store = {}
    for col, smooth in [('place_name',5),('district',10),('soil_type',10),
                        ('landcover',10),('road_quality',10),('flood_occurrence_current_event',10)]:
        tr_te, te_te = fold_safe_te(X, y, X_test, col, smooth, gm, SEED)
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
        ('historical_flood_count',f_tr,f_te,'fl_te_x_flood'),
        ('rainfall_7d_mm',       s_tr, s_te, 'soil_te_x_rain'),
    ]:
        X[col_name]      = tr_v * X[feat]
        X_test[col_name] = te_v * X_test[feat]

    X_arr      = X.values.astype(np.float32)
    X_test_arr = X_test.values.astype(np.float32)
    y_logit_train = y.values.astype(np.float32)
    y_logit_pseudo = y_pseudo.values.astype(np.float32)
    y_true_train = all_data.iloc[:N_TRAIN][TARGET].values.astype(np.float32)
    np.savez(TE_CACHE_FILE, X_arr=X_arr, X_test_arr=X_test_arr, 
             y_logit_train=y_logit_train, y_logit_pseudo=y_logit_pseudo, y_true_train=y_true_train)
    print(f"Features engineered and cached! Count: {X_arr.shape[1]}")


# ─── 60-Fold Training Engine with Pseudo-Labeling ─────────────────────
rkf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
TOTAL_FOLDS = N_FOLDS * N_REPEATS

def train_and_cache(name, train_fn):
    oof_path  = os.path.join(CHECKPOINT_DIR, f"{name}_oof.npy")
    test_path = os.path.join(CHECKPOINT_DIR, f"{name}_test.npy")
    
    if os.path.exists(oof_path) and os.path.exists(test_path):
        print(f"[SKIP] Model '{name}' already complete. Checkpoint loaded.")
        return
    
    print(f"\n>>> Starting Model: {name} ({TOTAL_FOLDS}-Folds)")
    start_time = time.time()
    
    # We must average out the repeated predictions
    oof_preds_sum  = np.zeros(len(X_arr))
    oof_counts     = np.zeros(len(X_arr))
    test_preds     = np.zeros(len(X_test_arr))
    
    for fold, (tr_i, val_i) in enumerate(rkf.split(X_arr)):
        # 1. Base training set
        X_tr, y_tr = X_arr[tr_i], y_logit_train[tr_i]
        
        # 2. Append Pseudo-Labels to training set!
        X_tr_full = np.vstack((X_tr, X_test_arr))
        y_tr_full = np.concatenate((y_tr, y_logit_pseudo))
        
        # 3. Validation set (STRICTLY original train data, no pseudo labels)
        X_va, y_va = X_arr[val_i], y_logit_train[val_i]
        y_va_true  = y_true_train[val_i]  # For accurate RMSE printing
        
        # Train on transformed targets
        preds_va_logit, preds_te_logit = train_fn(X_tr_full, y_tr_full, X_va, y_va, X_test_arr)
        
        # Convert predictions back to probability scale [0, 1]
        preds_va_prob = to_prob(preds_va_logit)
        preds_te_prob = to_prob(preds_te_logit)
        
        oof_preds_sum[val_i] += preds_va_prob
        oof_counts[val_i]    += 1
        test_preds           += preds_te_prob / TOTAL_FOLDS
        
        if (fold + 1) % N_FOLDS == 0 or fold == 0:
            print(f"  Fold {fold+1}/{TOTAL_FOLDS} | Prob RMSE = {np.sqrt(mean_squared_error(y_va_true, preds_va_prob)):.5f}")
        
    final_oof = oof_preds_sum / oof_counts
    rmse_total = np.sqrt(mean_squared_error(y_true_train, final_oof))
    mins = (time.time() - start_time) / 60
    print(f"<<< Finished Model: {name} | Total RMSE = {rmse_total:.5f} | Time = {mins:.1f} min")
    
    np.save(oof_path, final_oof)
    np.save(test_path, test_preds)

# ─── Define Model Trainers (Optimizing on Logit Scale) ────────────────
def get_lgb_trainer(params):
    def train_fn(X_tr, y_tr, X_va, y_va, X_te):
        m = lgb.train(params, lgb.Dataset(X_tr, y_tr),
                      5000, valid_sets=[lgb.Dataset(X_va, y_va)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        return m.predict(X_va), m.predict(X_te)
    return train_fn

def get_cat_trainer(params):
    def train_fn(X_tr, y_tr, X_va, y_va, X_te):
        m = CatBoostRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=0)
        return m.predict(X_va), m.predict(X_te)
    return train_fn

def get_xgb_trainer(params):
    def train_fn(X_tr, y_tr, X_va, y_va, X_te):
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        dte = xgb.DMatrix(X_te)
        m = xgb.train(params, dtr, num_boost_round=5000, 
                      evals=[(dva, 'val')], early_stopping_rounds=150, verbose_eval=False)
        return m.predict(dva), m.predict(dte)
    return train_fn

# ─── Execute Grid ───────────────────────────────────
print("\n=== Commencing V19 Excalibur Model Training Queue ===")

# LightGBM
p_rmse = {'objective':'regression','metric':'rmse','learning_rate':0.03,
         'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,
         'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0,
         'n_jobs':-1,'verbose':-1,'seed':SEED}
train_and_cache("lgb_rmse", get_lgb_trainer(p_rmse))

p_deep = p_rmse.copy(); p_deep.update({'num_leaves':255, 'min_child_samples':30, 'learning_rate':0.02})
train_and_cache("lgb_deep", get_lgb_trainer(p_deep))

p_dart = p_rmse.copy(); p_dart.update({'boosting_type': 'dart', 'learning_rate': 0.05})
train_and_cache("lgb_dart", get_lgb_trainer(p_dart))

# XGBoost
if HAS_XGB:
    p_xgb_sq = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'learning_rate': 0.03,
                'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.7, 
                'alpha': 0.1, 'lambda': 1.0, 'seed': SEED, 'tree_method': 'hist'}
    train_and_cache("xgb_sq", get_xgb_trainer(p_xgb_sq))
    
    p_xgb_hub = p_xgb_sq.copy(); p_xgb_hub['objective'] = 'reg:pseudohubererror'
    train_and_cache("xgb_huber", get_xgb_trainer(p_xgb_hub))

# CatBoost Deep Sweep
for d in [4, 6, 8, 10]:
    p_cat = dict(iterations=5000, learning_rate=0.03, depth=d,
                 l2_leaf_reg=3.0, min_data_in_leaf=15,
                 subsample=0.8, colsample_bylevel=0.7,
                 random_seed=SEED, task_type='CPU', verbose=0,
                 eval_metric='RMSE', early_stopping_rounds=150)
    train_and_cache(f"cat_d{d}", get_cat_trainer(p_cat))

print("\n=== All V19 Excalibur Models Completed ===")
print("Run `v19_stacker.py` to compile the final submission!")
