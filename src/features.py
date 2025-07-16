import pandas as pd

def load_processed_data(file_path="data/processed_nyc_taxi.csv"):
    """Load processed NYC taxi data from a CSV file.
    Args:
        file_path (str): Path to the CSV file containing processed data.
    Returns:
        pd.DataFrame: DataFrame containing the processed NYC taxi data."""
    df = pd.read_csv(file_path)
    return df

def get_feature_target_split(df, target_column):
    """Split DataFrame into features and target variable.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column to predict.
    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series."""
    features = [
        "trip_distance", "pickup_hour", "pickup_dayofweek",
        "is_weekend", "is_rush_hour", "pickup_hour_sin", "pickup_hour_cos"
    ]
    X = df[features]
    y = df[target_column]
    return X, y
