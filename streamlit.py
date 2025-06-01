import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.title("Crop Loss Prediction Dashboard")

st.write("This dashboard displays crop loss predictions.")

# Generate dummy data
data = {
    'NDVI': np.random.rand(100),
    'NDWI': np.random.rand(100),
    'NDCSI': np.random.rand(100),
    'crop_loss': np.random.rand(100)
}

df = pd.DataFrame(data)

# Display the DataFrame
st.dataframe(df)

# Initialize Folium map
m = folium.Map(location=[51.5, 7.5], zoom_start=8)

# Add a marker to the map
folium.Marker([51.5, 7.5], popup="Crop Loss Area").add_to(m)

# Render the map using st_folium
st_data = st_folium(m, width=700, height=500)

# Add more features here
