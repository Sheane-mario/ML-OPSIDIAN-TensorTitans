"""Quick EDA to understand the dataset"""
import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print("=== TRAIN SHAPE ===")
print(train.shape)

print("\n=== ALL COLUMNS ===")
for col in train.columns:
    dtype = train[col].dtype
    nuniq = train[col].nunique()
    miss  = train[col].isnull().sum()
    print(f"  {col:45s} dtype={str(dtype):10s} nunique={nuniq:6d} missing={miss}")

print("\n=== TARGET DISTRIBUTION ===")
print(train['flood_risk_score'].describe())
print("\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {train['flood_risk_score'].quantile(p/100):.4f}")

print("\n=== DISTRICT DISTRIBUTION ===")
print(train['district'].value_counts().head(30))

print("\n=== CATEGORICAL VALUE COUNTS ===")
cats = ['landcover', 'soil_type', 'water_supply', 'electricity', 
        'road_quality', 'urban_rural', 'water_presence_flag', 
        'flood_occurrence_current_event']
for c in cats:
    print(f"\n{c}:")
    print(train[c].value_counts())

print("\n=== NUMERICAL STATS ===")
nums = ['elevation_m', 'distance_to_river_m', 'population_density_per_km2',
        'built_up_percent', 'rainfall_7d_mm', 'monthly_rainfall_mm',
        'drainage_index', 'ndvi', 'ndwi', 'historical_flood_count',
        'infrastructure_score', 'nearest_hospital_km', 'nearest_evac_km',
        'seasonal_index', 'terrain_roughness_index', 'socioeconomic_status_index',
        'extreme_weather_index']
print(train[nums].describe().round(3).to_string())

print("\n=== CORRELATION WITH TARGET (top) ===")
numeric_train = train.select_dtypes(include=[np.number])
corr = numeric_train.corr()['flood_risk_score'].abs().sort_values(ascending=False)
print(corr.head(20))
