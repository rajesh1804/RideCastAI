import os
import requests

# Ensure data/ directory exists
os.makedirs("data", exist_ok=True)

url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
output_path = "data/yellow_tripdata_2023-01.parquet"

print("Downloading dataset...")
r = requests.get(url)

with open(output_path, "wb") as f:
    f.write(r.content)

print(f"Dataset downloaded successfully to {output_path}")
