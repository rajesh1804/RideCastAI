import streamlit as st
from datetime import datetime
import plotly.express as px
import pandas as pd
from utils import predict_eta_fare
from pathlib import Path

st.set_page_config(page_title="RideCastAI", layout="wide")
st.title("🚖 RideCastAI: ETA & Fare Estimator")

st.markdown("""
Welcome to **RideCastAI** – your intelligent ride-hailing assistant.  
Enter trip details and get instant predictions for:
- Estimated Time of Arrival (ETA)
- Dynamic Fare (with confidence range)
- NYC demand heatmap
""")

# ------------------------------------
# 📍 Trip Input Form
# ------------------------------------
st.header("📋 Trip Input")

with st.sidebar.expander("📈 Model Evaluation Metrics"):
    try:
        eta_metrics = Path("src/models/eta_metrics.txt").read_text()
        fare_metrics = Path("src/models/fare_metrics.txt").read_text()
        st.markdown("#### ETA Model (MAE)")
        st.code(eta_metrics)
        st.markdown("#### Fare Model (MAE)")
        st.code(fare_metrics)
    except:
        st.warning("⚠️ Metrics not available. Run training scripts again.")

col1, col2 = st.columns(2)
with col1:
    pickup_hour = st.slider("Pickup Hour", 0, 23, value=datetime.now().hour)
with col2:
    pickup_day = st.selectbox("Pickup Day", options=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    pickup_dayofweek = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(pickup_day)

trip_distance = st.slider("Trip Distance (miles)", 0.1, 25.0, 5.0, step=0.1)

# ------------------------------------
# 🚀 Prediction
# ------------------------------------
if st.button("🔍 Predict ETA & Fare"):
    eta, fare, fare_low, fare_high, duration  = predict_eta_fare(pickup_hour, pickup_dayofweek, trip_distance)

    st.subheader("📊 Prediction Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("🕒 Estimated ETA", f"{eta:.2f} mins")
    col2.metric("💰 Predicted Fare", f"${fare}")
    col3.metric("↔️ Fare Range", f"${fare_low} - ${fare_high}")
    st.caption(f"🧠 Prediction completed in {duration} ms")

# ------------------------------------
# 🗺️ Heatmap Section
# ------------------------------------
st.header("🔥 NYC Ride Demand Heatmap")

@st.cache_data
def load_heatmap_data():
    import numpy as np
    np.random.seed(42)
    df = pd.DataFrame({
        "lat": np.random.uniform(40.6, 40.9, 100),
        "lon": np.random.uniform(-74.05, -73.75, 100),
        "demand": np.random.randint(5, 100, 100)
    })
    return df

heatmap_df = load_heatmap_data()

fig = px.density_map(
    heatmap_df,
    lat="lat",
    lon="lon",
    z="demand",
    radius=10,
    center=dict(lat=40.75, lon=-73.9),
    zoom=10,
    map_style="carto-positron",
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built by Rajesh •")
