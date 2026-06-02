import pandas as pd, numpy as np
train = pd.read_csv('data/train.csv')
TARGET = 'flood_risk_score'

# Within-place residuals
place_means = train.groupby('place_name')[TARGET].mean()
train['place_mean'] = train['place_name'].map(place_means)
train['residual'] = train[TARGET] - train['place_mean']
print('=== WITHIN-PLACE RESIDUAL STATS ===')
print('Residual std:', round(train['residual'].std(), 4))
print('Residual range:', round(train['residual'].min(),4), 'to', round(train['residual'].max(),4))

# Feature correlations with residuals
num_cols = ['rainfall_7d_mm','monthly_rainfall_mm','drainage_index','ndvi','ndwi',
            'historical_flood_count','infrastructure_score','elevation_m',
            'distance_to_river_m','built_up_percent','population_density_per_km2',
            'extreme_weather_index','seasonal_index','terrain_roughness_index',
            'socioeconomic_status_index','inundation_area_sqm',
            'rainfall_7d_mm_log1p','distance_to_river_m_log1p','ndwi_qmap','ndvi_qmap']

print('\n=== FEATURE CORRELATIONS WITH WITHIN-PLACE RESIDUAL ===')
corrs = [(c, train['residual'].corr(train[c])) for c in num_cols if c in train.columns]
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for c, r in corrs:
    print(f'  {c:45s} {r:+.4f}')

# flood_occurrence_current_event
print('\n=== FLOOD OCCURRENCE vs TARGET ===')
print(train.groupby('flood_occurrence_current_event')[TARGET].agg(['mean','std','count']))

print('\n=== FLOOD OCCURRENCE vs RESIDUAL ===')
print(train.groupby('flood_occurrence_current_event')['residual'].agg(['mean','std','count']))

# water_presence_flag
print('\n=== WATER_PRESENCE_FLAG vs TARGET ===')
print(train.groupby('water_presence_flag')[TARGET].agg(['mean','std','count']))

# is_synthetic
print('\n=== IS_SYNTHETIC ===')
print(train['is_synthetic'].value_counts())

# Check if is_synthetic varies
print('\n=== GENERATION_DATE month vs residual corr ===')
train['gen_month'] = pd.to_datetime(train['generation_date']).dt.month
print('month corr with residual:', round(train['residual'].corr(train['gen_month']),4))
print('month corr with target:', round(train[TARGET].corr(train['gen_month']),4))
