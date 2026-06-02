import pandas as pd
import os

SUB_DIR = "submissions"

# Our new best submission is V15 Huber Stack (0.38189)
# Our second best is V11 Stack (0.38194)
# Our third best is V14 Weighted Ensemble (0.38195)

try:
    v15 = pd.read_csv(os.path.join(SUB_DIR, "submission_v15_huber_stack.csv"))
    v11 = pd.read_csv(os.path.join(SUB_DIR, "submission_v11.csv"))
    v14w = pd.read_csv(os.path.join(SUB_DIR, "submission_v14_weighted_ensemble.csv"))
    
    # 1. Blend V15 (Huber) and V14w (Meta-Ensemble)
    # The meta-ensemble is incredibly stable (3-model blend). V15 is our best single ensemble.
    sub_blend = v15.copy()
    sub_blend['flood_risk_score'] = (v15['flood_risk_score'] * 0.5) + (v14w['flood_risk_score'] * 0.5)
    sub_blend.to_csv("submission_v16_meta_blend_1.csv", index=False)
    print("Created submission_v16_meta_blend_1.csv (V15 50% + V14w 50%)")
    
    # 2. Weighted blend of the 3 top performers
    sub_weighted = v15.copy()
    sub_weighted['flood_risk_score'] = (
        v15['flood_risk_score'] * 0.5 + 
        v11['flood_risk_score'] * 0.3 + 
        v14w['flood_risk_score'] * 0.2
    )
    sub_weighted.to_csv("submission_v16_meta_blend_2.csv", index=False)
    print("Created submission_v16_meta_blend_2.csv (V15 50% + V11 30% + V14w 20%)")

except Exception as e:
    print("Error blending:", e)
