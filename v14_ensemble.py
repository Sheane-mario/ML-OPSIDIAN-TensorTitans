import pandas as pd
import os

SUB_DIR = "submissions"

# Top 3 submissions based on LB scores
# V11: 0.38194 (Best)
# V13: 0.38205
# V8:  0.38215

try:
    v11 = pd.read_csv(os.path.join(SUB_DIR, "submission_v11.csv"))
    v13 = pd.read_csv(os.path.join(SUB_DIR, "submission_v13.csv"))
    v8  = pd.read_csv(os.path.join(SUB_DIR, "submission_v8.csv"))
    
    # 1. Simple Average of Top 3
    sub_mean = v11.copy()
    sub_mean['flood_risk_score'] = (v11['flood_risk_score'] + v13['flood_risk_score'] + v8['flood_risk_score']) / 3.0
    sub_mean.to_csv("submission_v14_mean_ensemble.csv", index=False)
    print("Created submission_v14_mean_ensemble.csv")
    
    # 2. Weighted Average (heavier weight to better models)
    sub_weighted = v11.copy()
    # Weights: V11 (50%), V13 (30%), V8 (20%)
    sub_weighted['flood_risk_score'] = (
        v11['flood_risk_score'] * 0.5 + 
        v13['flood_risk_score'] * 0.3 + 
        v8['flood_risk_score'] * 0.2
    )
    sub_weighted.to_csv("submission_v14_weighted_ensemble.csv", index=False)
    print("Created submission_v14_weighted_ensemble.csv")

except Exception as e:
    print("Error blending:", e)
