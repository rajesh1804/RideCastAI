from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from features import load_processed_data, get_feature_target_split
import os
import time
import numpy as np

# Fix RNG state for backward compatibility
np.random.bit_generator = None


print("📥 Loading processed data...")
df = load_processed_data()
X, y = get_feature_target_split(df, target_column="trip_duration")

print("🔀 Splitting into train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Starting ETA model training...")
start_time = time.time()

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

end_time = time.time()
duration = end_time - start_time
print(f"✅ Training completed in {duration:.2f} seconds")

print("📈 Evaluating model on test set...")
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"📊 ETA MAE: {mae:.2f} minutes")

# Save MAE to file for UI
os.makedirs("src/models", exist_ok=True)
with open("src/models/eta_metrics.txt", "w") as f:
    f.write(f"{mae:.2f} minutes\n")

joblib.dump(model, "src/models/eta_model.pkl")
print("💾 ETA model saved to src/models/eta_model.pkl")
