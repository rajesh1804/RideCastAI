from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from features import load_processed_data, get_feature_target_split
import time
import numpy as np

# Fix RNG state for backward compatibility
np.random.bit_generator = None


print("🔄 Loading and preprocessing data...")
start = time.time()
df = load_processed_data()
X, y = get_feature_target_split(df, target_column="fare_amount")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data loaded and split. Time taken: {time.time() - start:.2f}s")

models = {}
quantiles = [0.1, 0.5, 0.9]

for q in quantiles:
    print(f"\n🚀 Training quantile model for q={q}...")
    q_start = time.time()

    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=q,
        n_estimators=100,      
        max_depth=5,          
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    models[q] = model

    print(f"✅ Done. MAE @ quantile {q}: {mae:.2f} | Time taken: {time.time() - q_start:.2f}s")

    model_path = f"src/models/fare_model_q{int(q*100)}.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

with open("src/models/fare_metrics.txt", "w") as f:
    for q in quantiles:
        preds = models[q].predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        f.write(f"Quantile {q}: {mae:.2f}\n")

print("\n🎉 All quantile models trained and saved successfully.")
