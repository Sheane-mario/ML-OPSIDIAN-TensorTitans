"""
V18 Titan Stacker

Reads all available checkpoints from the Titan training script, 
dynamically compiles them into a stacking matrix, and uses our proven 
HuberRegressor to generate the ultimate submission.

You can run this at any time, even if V18 Titan is still training! It will 
simply stack whatever models have finished so far.
"""

import numpy as np
import pandas as pd
import os
import glob
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score

CHECKPOINT_DIR = "titan_checkpoints"
DATA_DIR = "data"
OUTPUT_FILE = "submission_v18_titan.csv"

print("=== V18 Titan Auto-Stacker ===")

oof_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*_oof.npy")))
if not oof_files:
    print("No checkpoints found! Have you run `v18_titan_solution.py` yet?")
    exit(1)

model_names = [os.path.basename(f).replace("_oof.npy", "") for f in oof_files]
print(f"Found {len(model_names)} completed models:")
for m in model_names:
    print(f"  - {m}")

print("\nLoading checkpoints...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
y_arr = train['flood_risk_score'].values
test_record_ids = test['record_id'].values

oof_stack = []
test_stack = []

for m in model_names:
    oof_stack.append(np.load(os.path.join(CHECKPOINT_DIR, f"{m}_oof.npy")))
    test_stack.append(np.load(os.path.join(CHECKPOINT_DIR, f"{m}_test.npy")))

oof_stack = np.column_stack(oof_stack)
test_stack = np.column_stack(test_stack)

print(f"Stacking matrix shape: {oof_stack.shape}")

print("\nTraining HuberRegressor (Epsilon=1.35) ...")
huber = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000)
huber.fit(oof_stack, y_arr)

huber_oof = huber.predict(oof_stack)
huber_test = huber.predict(test_stack)

rmse = np.sqrt(mean_squared_error(y_arr, huber_oof))
r2 = r2_score(y_arr, huber_oof)

print("\n=== Model Weights ===")
for i, name in enumerate(model_names):
    print(f"  {name:15s} : {huber.coef_[i]:.4f}")

print(f"\n[TITAN STACK] OOF RMSE = {rmse:.5f} | R2 = {r2:.4f}")

# Generate Submission
preds_clipped = np.clip(huber_test, 0, 1)
sub = pd.DataFrame({
    'record_id': test_record_ids,
    'flood_risk_score': preds_clipped
})

sub.to_csv(OUTPUT_FILE, index=False)
print(f"\n[OK] Submission saved to {OUTPUT_FILE}")
