import ee
import ee.mapclient
import sklearn
from sklearn.ensemble import CaboostRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Authenticate with Google Earth Engine
try:
    ee.Initialize()
    print('Earth Engine initialized successfully.')
except Exception as e:
    print(f'Failed to initialize Earth Engine: {e}')
    try:
        ee.Authenticate()
        ee.Initialize()
    except Exception as e:
        print(f'Failed to authenticate Earth Engine: {e}')

# Define region of interest (Nord Rhine-Westphalia)
roi = ee.Geometry.Polygon(
        [[[6.0, 50.5],
          [9.0, 50.5],
          [9.0, 52.5],
          [6.0, 52.5]]])

# Define time range
start_date = '2020-01-01'
end_date = '2021-12-31'

# Load Sentinel-2 data
sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR').filterDate(start_date, end_date).filterBounds(roi)

# Function to calculate NDVI
def calculate_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Function to calculate NDWI
def calculate_ndwi(image):
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands(ndvi)

# Function to calculate NDCSI (Clay Minerals Ratio)
def calculate_ndcsi(image):
    ndcsi = image.normalizedDifference(['B11', 'B12']).rename('NDCSI')
    return image.addBands(ndcsi)

# Apply functions to Sentinel-2 data
sentinel2_ndvi = sentinel2.map(calculate_ndvi)
sentinel2_ndwi = sentinel2.map(calculate_ndwi)
sentinel2_ndcsi = sentinel2.map(calculate_ndcsi)

# Function to extract features and labels
def extract_features(image):
    ndvi = image.select('NDVI').mean()
    ndwi = image.select('NDWI').mean()
    ndcsi = image.select('NDCSI').mean()
    # Replace 'crop_loss' with the actual band name containing crop loss data
    crop_loss = image.select('B2').mean()  # Example: using blue band as placeholder for crop loss
    return ee.Feature(None, {
        'NDVI': ndvi,
        'NDWI': ndwi,
        'NDCSI': ndcsi,
        'crop_loss': crop_loss
    })

# Prepare training data
training_data = sentinel2_ndvi.map(extract_features).getInfo()

# Convert training data to pandas DataFrame
features = []
labels = []
for item in training_data['features']:
    properties = item['properties']
    features.append([properties['NDVI'], properties['NDWI'], properties['NDCSI']])
    labels.append(properties['crop_loss'])

df = pd.DataFrame(features, columns=['NDVI', 'NDWI', 'NDCSI'])
df['crop_loss'] = labels

# Prepare data for LightGBM
X = df[['NDVI', 'NDWI', 'NDCSI']]
y = np.log1p(df['crop_loss'].values)  # Use log1p transformation

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)


# LightGBM Model Training
def light_gbm_model_run(train_x, train_y, validation_x, validation_y):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 100,
        "learning_rate" : 0.001,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }

    lg_train = lgb.Dataset(train_x, label=train_y)
    lg_validation = lgb.Dataset(validation_x, label=validation_y)
    evals_result_lgbm = {}

    model_light_gbm = lgb.train(params, lg_train, 5000,
                      valid_sets=[lg_train, lg_validation],
                      early_stopping_rounds=100,
                      verbose_eval=150,
                      evals_result=evals_result_lgbm )

    pred_test_light_gbm = np.expm1(model_light_gbm.predict(X_validation, num_iteration=model_light_gbm.best_iteration )) # Corrected prediction

    return pred_test_light_gbm, model_light_gbm, evals_result_lgbm

# Training and output of LightGBM Model
predictions_test_y_light_gbm, model_lgbm, evals_result = light_gbm_model_run(X_train, y_train, X_validation, y_validation)
print('Output of LightGBM Model training..')

# Print model details
print('Trained LightGBM model:', model_lgbm)

# Evaluation (example)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(np.expm1(y_validation), predictions_test_y_light_gbm)
print('Mean Squared Error:', mse)

print('Script completed.')
