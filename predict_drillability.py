import joblib
import pandas as pd

# 1. Load the pre-trained SVR pipeline (includes scaler + model)
model = joblib.load('best_rock_drillability_svr.pkl')

# 2. Input new laboratory rock sample data (features must match dataset order)
# Example: 2 new rock samples with 6 mechanical features each
new_rock_samples = pd.DataFrame([
    [189.8, 23.4, 49.8, 8.6, 4588.6, 2.76],
    [78.2,  6.6,  22.9, 4.4, 3405.5, 2.38]
])

# 3. Generate DRI Predictions
predictions = model.predict(new_rock_samples)

print("--- Rock Drillability Index (DRI) Predictions ---")
for i, pred in enumerate(predictions, 1):
    print(f"Sample {i}: Predicted DRI = {pred:.2f}")