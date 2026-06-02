import pandas as pd
import os

SUB_DIR = "submissions"

# Load the ultimate models
try:
    v15 = pd.read_csv(os.path.join(SUB_DIR, "submission_v15_huber_stack.csv"))
    v11 = pd.read_csv(os.path.join(SUB_DIR, "submission_v11.csv"))
    v17 = pd.read_csv(os.path.join("submission_v17.csv"))  # V17 writes to current dir

    # Genesis Ultimate Blend
    # 40% V15 (Our best proven LB model)
    # 30% V11 (Our second best LB model)
    # 30% V17 (Our new 20-fold mega-ensemble)
    
    sub_ultimate = v15.copy()
    sub_ultimate['flood_risk_score'] = (
        v15['flood_risk_score'] * 0.40 + 
        v11['flood_risk_score'] * 0.30 + 
        v17['flood_risk_score'] * 0.30
    )
    sub_ultimate.to_csv(os.path.join(SUB_DIR, "submission_v17_genesis_ultimate.csv"), index=False)
    print("Created submission_v17_genesis_ultimate.csv")
    
    # Just in case, also create a simple average of the three
    sub_mean = v15.copy()
    sub_mean['flood_risk_score'] = (
        v15['flood_risk_score'] + 
        v11['flood_risk_score'] + 
        v17['flood_risk_score']
    ) / 3.0
    sub_mean.to_csv(os.path.join(SUB_DIR, "submission_v17_genesis_mean.csv"), index=False)
    print("Created submission_v17_genesis_mean.csv")

except Exception as e:
    print("Error blending:", e)
