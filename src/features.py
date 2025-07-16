import pandas as pd

def load_processed_data(file_path="data/processed_nyc_taxi.csv"):
    df = pd.read_csv(file_path)
    return df

def get_feature_target_split(df, target_column):
    features = [
        "trip_distance", "pickup_hour", "pickup_dayofweek",
        "is_weekend", "is_rush_hour", "pickup_hour_sin", "pickup_hour_cos"
    ]
    X = df[features]
    y = df[target_column]
    return X, y
