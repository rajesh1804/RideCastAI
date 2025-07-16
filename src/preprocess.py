# src/preprocess.py
import pandas as pd
from tqdm import tqdm
from datetime import datetime

tqdm.pandas()

INPUT_FILE = "data/yellow_tripdata_2023-01.parquet"
OUTPUT_FILE = "data/processed_nyc_taxi.csv"

def load_data():
    """Load NYC taxi data from a Parquet file.
    Returns:
        pd.DataFrame: DataFrame containing the raw NYC taxi data."""
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")
    return df

def clean_data(df):
    """Clean the NYC taxi data.
    Args:
        df (pd.DataFrame): DataFrame containing the raw NYC taxi data.
    Returns:
        pd.DataFrame: Cleaned DataFrame with necessary columns and no missing values."""
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount"])
    df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]

    # Compute trip duration in minutes
    df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

    # Filter out unrealistic trips
    df = df[(df["trip_duration"] > 1) & (df["trip_duration"] < 120)]

    return df

def feature_engineering(df):
    """Perform feature engineering on the NYC taxi data.
    Args:
        df (pd.DataFrame): DataFrame containing the cleaned NYC taxi data.
    Returns:
        pd.DataFrame: DataFrame with additional features for model training."""
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = df["pickup_dayofweek"].isin([5,6]).astype(int)
    df["is_rush_hour"] = df["pickup_hour"].isin([7,8,9,17,18,19]).astype(int)

    # Optional: for cyclical encoding
    import numpy as np
    df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"]/24)
    df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"]/24)

    return df

def save_processed_data(df):
    """Save the processed NYC taxi data to a CSV file.
    Args:
        df (pd.DataFrame): DataFrame containing the processed NYC taxi data.
    """
    columns_to_save = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID",
        "trip_distance", "fare_amount", "trip_duration",
        "pickup_hour", "pickup_dayofweek", "is_weekend", "is_rush_hour",
        "pickup_hour_sin", "pickup_hour_cos"
    ]
    df[columns_to_save].to_csv(OUTPUT_FILE, index=False)
    print(f"Processed data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    save_processed_data(df)
