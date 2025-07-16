import joblib
import numpy as np
import time
import urllib.request
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

MODEL_DIR = "src/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_URL = os.getenv("MODEL_BASE_URL")

def download_model(filename):
    """Download a model file from the specified URL if it doesn't exist locally.
    Args:
        filename (str): Name of the model file to download.
    Returns:
        str: Local path to the downloaded model file."""
    url = f"{BASE_URL}/{filename}"
    dest_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(dest_path):
        print(f"📥 Downloading {filename} from GitHub...")
        urllib.request.urlretrieve(url, dest_path)
    return dest_path

# Download if needed
eta_model_path = download_model("eta_model.pkl")
fare_q10_path = download_model("fare_model_q10.pkl")
fare_q50_path = download_model("fare_model_q50.pkl")
fare_q90_path = download_model("fare_model_q90.pkl")

# Load all models
eta_model = joblib.load(eta_model_path)
fare_model_q10 = joblib.load(fare_q10_path)
fare_model_q50 = joblib.load(fare_q50_path)
fare_model_q90 = joblib.load(fare_q90_path)

def extract_features(hour, dayofweek, distance):
    """Extract features for model prediction.
    Args:
        hour (int): Hour of the day (0-23).
        dayofweek (int): Day of the week (0=Monday, 6=Sunday).
        distance (float): Distance in kilometers.
    Returns:
        List of features for prediction."""
    is_weekend = int(dayofweek >= 5)
    is_rush = int(hour in [7, 8, 9, 17, 18, 19])
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    return pd.DataFrame([{
        "trip_distance": distance,
        "pickup_hour": hour,
        "pickup_dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush,
        "pickup_hour_sin": hour_sin,
        "pickup_hour_cos": hour_cos
    }])

def predict_eta_fare(hour, dayofweek, distance):
    """Predict ETA and fare based on input features.
    Args:
        hour (int): Hour of the day (0-23).
        dayofweek (int): Day of the week (0=Monday, 6=Sunday).
        distance (float): Distance in kilometers.
    Returns:
        Tuple of predicted ETA, fare (low, mid, high), and duration of prediction in milliseconds."""
    start = time.time()

    X = extract_features(hour, dayofweek, distance)
    eta = eta_model.predict(X)[0]
    fare_low = fare_model_q10.predict(X)[0]
    fare_mid = fare_model_q50.predict(X)[0]
    fare_high = fare_model_q90.predict(X)[0]

    duration = round((time.time() - start) * 1000, 2)  # milliseconds
    return round(eta, 2), round(fare_mid, 2), round(fare_low, 2), round(fare_high, 2), duration
