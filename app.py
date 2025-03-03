import json
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from google.cloud import bigquery
from google.oauth2 import service_account

# ---- Streamlit Page Configuration ----
st.set_page_config(page_title="San Jose Traffic Dashboard", layout="wide")

# ---- App Title ----
st.title("🚦 San Jose Traffic Dashboard")

# ---- Load Google Cloud Credentials ----
try:
    service_account_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = bigquery.Client(credentials=credentials, project=service_account_info["project_id"])
    st.write("✅ Successfully connected to BigQuery!")
except Exception as e:
    st.error(f"🚨 Error loading Google Cloud credentials: {e}")

# ---- Load Data from BigQuery ----
PROJECT_ID = "averagetraffic"
TABLE_NAME = "Traffic_Data.Cleaned_SJ_Traffic"

query = f"SELECT * FROM `{PROJECT_ID}.{TABLE_NAME}`"
df = client.query(query).to_dataframe()

# ---- Sidebar for Filtering ----
st.sidebar.header("Filter Data")
selected_street = st.sidebar.selectbox("Select a Street", ["All"] + list(df["STREETONE"].unique()))

# Apply Filter
if selected_street != "All":
    df = df[df["STREETONE"] == selected_street]

# ---- Display Data ----
st.write("### Traffic Data", df)

# ---- Create Interactive Map ----
st.write("### Traffic Map of San Jose")
traffic_map = folium.Map(location=[df["LATITUDE"].mean(), df["LONGITUDE"].mean()], zoom_start=12)

# Add markers
for _, row in df.iterrows():
    folium.Marker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        popup=f"Street: {row['STREETONE']}<br>ADT: {row['ADT']}",
        icon=folium.Icon(color="red")
    ).add_to(traffic_map)

# Display Map
folium_static(traffic_map)

# ---- Data Visualizations ----
st.write("### Top 10 Busiest Roads by ADT")
top_streets = df.groupby("STREETONE")["ADT"].mean().nlargest(10)
st.bar_chart(top_streets)

st.write("### Traffic Volume Over Time")
traffic_over_time = df.groupby("COUNTDATE")["ADT"].mean()
st.line_chart(traffic_over_time)

# ---- Embed Looker Studio Dashboard ----
st.write("## 📊 Looker Studio Dashboard")
LOOKER_STUDIO_EMBED_URL = "https://lookerstudio.google.com/embed/reporting/14d0f77c-e681-4a0f-ba3f-b7051d514f34/page/NdI4E"
st.components.v1.iframe(LOOKER_STUDIO_EMBED_URL, width=900, height=600, scrolling=True)

# ---- ADDING MACHINE LEARNING PREDICTION ----
st.write("## 🚀 Predict Traffic Volume using BigQuery ML")

# Get user input for Latitude & Longitude
latitude = st.number_input("Enter Latitude", value=37.3382)
longitude = st.number_input("Enter Longitude", value=-121.8863)
facility_id = st.number_input("Enter Facility ID", value=12345)
intid = st.number_input("Enter INTID", value=6789)

# Run ML.PREDICT Query when user clicks button
if st.button("Predict ADT Traffic Volume"):
    ml_query = f"""
    SELECT predicted_ADT
    FROM ML.PREDICT(
        MODEL `averagetraffic.Traffic_Data.traffic_prediction_model`,
        (SELECT {latitude} AS LATITUDE, {longitude} AS LONGITUDE, 
                {facility_id} AS FACILITYID, {intid} AS INTID)
    )
    """
    
    try:
        ml_result = client.query(ml_query).to_dataframe()
        
        if not ml_result.empty:
            st.success(f"Predicted ADT Traffic Volume: **{ml_result['predicted_ADT'].iloc[0]:.2f}**")
        else:
            st.error("No prediction available. Check input values!")

    except Exception as e:
        st.error(f"Error running prediction: {e}")
