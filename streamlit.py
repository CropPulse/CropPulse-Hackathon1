import streamlit as st
import pandas as pd
import numpy as np

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

# Display the map
st.write("Map will be displayed here")

# Initialize Folium map
m = folium.Map(location=[51.5, 7.5], zoom_start=8)

# Add a raster layer from Earth Engine
try:
    # Import the MODIS land cover collection.
    lc = ee.ImageCollection('MODIS/006/MCD12Q1')
    # Initial date of interest (inclusive).
    i_date = '2017-01-01'
    # Select one image
    lc_img = lc.select('LC_Type1').filterDate(i_date).first()

    # Add the Earth Engine layer to the Folium map.
    m.add_ee_layer(lc_img, {'min': 0, 'max': 17, 'palette': ['006633', 'E5FFCC', '33280D', 'F7E084', '6F6F6F', '9C9C9C', '1A5BAB', '67A9CF', 'C3D9DF', 'E0ECF4', '68B329', '99D98C', 'D0D1E6', 'A99274', 'D9D0C1', 'D5E0BB', 'F2F2F2']}, 'MODIS Land Cover')
except Exception as e:
    print(f"Failed to add Earth Engine layer: {e}")
    st.write(f"Failed to add Earth Engine layer: {e}")

# Render the map using st_folium
st_folium(m, width=700, height=500)

# Add more features here

import folium
from streamlit_folium import st_folium
import google.oauth2.service_account

# Authenticate with Google Earth Engine using the service account key
try:
    # Use the service account credentials from Streamlit secrets
    info = st.secrets["gee_service_account"]
    creds = google.oauth2.service_account.Credentials.from_service_account_info(info)
    ee.Initialize(creds=creds)
    print("Earth Engine authenticated successfully.")
except Exception as e:
    print(f"Failed to authenticate Earth Engine: {e}")

# Add Earth Engine drawing method to folium.
def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds a method for displaying Earth Engine image tiles to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer
