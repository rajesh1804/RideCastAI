import joblib
import numpy as np
import time

# Load all models
eta_model = joblib.load("src/models/eta_model.pkl")
fare_model_q10 = joblib.load("src/models/fare_model_q10.pkl")
fare_model_q50 = joblib.load("src/models/fare_model_q50.pkl")
fare_model_q90 = joblib.load("src/models/fare_model_q90.pkl")

def extract_features(hour, dayofweek, distance):
    is_weekend = int(dayofweek >= 5)
    is_rush = int(hour in [7, 8, 9, 17, 18, 19])
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    return [[
        distance, hour, dayofweek, is_weekend, is_rush, hour_sin, hour_cos
    ]]

def predict_eta_fare(hour, dayofweek, distance):
    start = time.time()

    X = extract_features(hour, dayofweek, distance)
    eta = eta_model.predict(X)[0]
    fare_low = fare_model_q10.predict(X)[0]
    fare_mid = fare_model_q50.predict(X)[0]
    fare_high = fare_model_q90.predict(X)[0]

    duration = round((time.time() - start) * 1000, 2)  # milliseconds
    return round(eta, 2), round(fare_mid, 2), round(fare_low, 2), round(fare_high, 2), duration
