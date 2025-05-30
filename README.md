# CropPulse-Hackathon1
﻿# Lightweight Geospatial ML Pipeline for Drought Stress Detection (Hackathon Guide)

## Introduction

In precision agriculture, satellite imagery and simple machine learning can help identify  **crop drought stress**  in near real-time. Industry tools like Ceres Imaging and Climate FieldView use aerial and satellite data to give growers field-level insights – for example, FieldView emphasizes early detection of risks such as drought stress[ceres.ai](https://ceres.ai/blog/ceres-ai-and-bayer-climate-fieldview-partner-to-empower-farm-operations-and-financial-stakeholders-with-ai-driven-insights#:~:text=For%20farmers%20and%20agribusiness%20operators%3A), and Ceres Imaging provides high-resolution maps of water stress, chlorophyll, and NDVI for each field[futurefarming.com](https://www.futurefarming.com/smart-farming/tools-data/ceres-imaging-measures-water-stress-at-plant-level/#:~:text=Key%20benefits%20that%20plant%20level,provides%20according%20to%20Ceres%20Imaging). In a hackathon setting, we can emulate some of these capabilities with an  **end-to-end pipeline**  built on open data and tools. This guide outlines a  **simple, reproducible workflow**  using  **Google Earth Engine (GEE)**  for data access and preprocessing, plus a lightweight ML model (e.g. Random Forest) to flag drought-stressed crops. We will work with a  **compact dataset**  (e.g. one 1-hectare field tile) to keep the process fast and within memory limits, while still producing insightful outputs like  **stress heatmaps**  or  **drought/no-drought flags**.

**What You Will Learn:**  We break down the pipeline into clear steps – from fetching satellite imagery and rainfall data, computing vegetation indices (NDVI, NDWI), masking clouds, to training a simple classifier. Along the way, we highlight code snippets (JavaScript/Python for GEE and scikit-learn/LightGBM) and link out to beginner-friendly tutorials for each stage. All components are designed to be  **space and compute efficient**  (under a few hundred MB of data, suitable for <1 GB RAM environments). By the end, you’ll have a small-scale but functional prototype for drought stress monitoring that can be expanded after the hackathon.

## Step 1: Satellite Data Collection and Preprocessing with GEE

**Selecting a Small Area of Interest:**  Choose a test field or tile (~1 hectare, ~100 m × 100 m) as the analysis area. This keeps data volumes low. Define an Area of Interest (AOI) in GEE, for example:

javascript

CopyEdit

`// Example AOI: a 1-ha rectangular field (coordinates in [lon, lat] format)  var region = ee.Geometry.Rectangle([30.0, -10.0, 30.001, -9.999]);` 

**Accessing Sentinel-2 Imagery:**  We use  **Sentinel-2**  surface reflectance data (10 m resolution) for vegetation indices. In Earth Engine, filter the image collection by date and location, and apply cloud masking. For instance, to get images over the field for a particular growing season (e.g. June–August 2021):

javascript

CopyEdit

`var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
            .filterBounds(region)
            .filterDate('2021-06-01', '2021-08-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50));` 

**Cloud Masking:**  Remove clouds to ensure indices aren’t skewed by cloud cover. A simple approach uses the Sentinel-2 QA60 bitmask band, where bits 10 and 11 indicate clouds/cirrus. We can define a mask function:

javascript

CopyEdit

`function  maskClouds(image) { var qa = image.select('QA60'); // Bits 10 and 11 are cloud and cirrus flag bits.  var mask = qa.bitwiseAnd(1 << 10).eq(0)
               .and(qa.bitwiseAnd(1 << 11).eq(0)); return image.updateMask(mask);
}` 

Apply this to each image:  `s2 = s2.map(maskClouds);`.  _(For more advanced cloud masking, see the  **s2cloudless**  tutorial from Google[developers.google.com](https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless#:~:text=This%20tutorial%20is%20an%20introduction,infrared%20%28NIR%29%20pixels), but the above QA60 mask is fast and sufficient for a hackathon.)_

**Calculating Vegetation Indices (NDVI & NDWI):**  Once we have cloud-free Sentinel-2 images, we compute indices that highlight crop health and water status. Two simple indices are:

-   **NDVI (Normalized Difference Vegetation Index):**  a well-known proxy for green biomass and vigor. NDVI is defined as  **(NIR – Red) / (NIR + Red)**[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi/#:~:text=The%20normalized%20difference%20vegetation%20index%2C,abbreviated%20NDVI%2C%20is%20defined%20as). For Sentinel-2, that is Band 8 (NIR) and Band 4 (red)[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi/#:~:text=%5C%5BNDVI%20%3A%3D%20%5Cmathtt%7BIndex%7D%28NIR%2CRED%29%20%3D%20%5Cfrac%7BNIR). NDVI ranges from -1 to 1; higher values (~0.5–0.8) mean lush vegetation, near zero indicates sparse/barren areas, and negative values indicate water[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi/#:~:text=The%20value%20range%20of%20the,1%5D%20for%20details). In Earth Engine, we can compute NDVI using the built-in  `normalizedDifference`  method for convenience[developers.google.com](https://developers.google.com/earth-engine/tutorials/tutorial_api_06#:~:text=match%20at%20L330%20var%20ndvi,B5%27%2C%20%27B4%27%5D%29.rename%28%27NDVI):
    
    javascript
    
    CopyEdit
    
    `var addNDVI = function(image) { var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI'); return image.addBands(ndvi);
    };
    s2 = s2.map(addNDVI);` 
    
-   **NDWI (Normalized Difference Water Index):**  an index sensitive to water content in the scene. By the common definition (McFeeters 1996), NDWI uses the green and NIR bands:  **(Green – NIR) / (Green + NIR)**[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/#:~:text=The%20NDWI%20is%20used%20to,was%20proposed%20by%20McFeeters%2C%201996). It highlights open water bodies (NDWI > 0.5 typically indicating water) and can differentiate water vs. vegetation[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/#:~:text=The%20NDWI%20is%20used%20to,was%20proposed%20by%20McFeeters%2C%201996)[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/#:~:text=Values%20description%3A%20Index%20values%20greater,2). We compute NDWI similarly:
    
    javascript
    
    CopyEdit
    
    `var addNDWI = function(image) { var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI'); return image.addBands(ndwi);
    };
    s2 = s2.map(addNDWI);` 
    
    _Note:_  Some literature uses  _“NDWI”_  to mean an index of vegetation water content using NIR and SWIR (also called NDMI). In this guide, we stick to the Green–NIR NDWI for water bodies[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/#:~:text=Note%3A%20NDWI%20index%20is%20often,NIR)  and acknowledge that a  **Normalized Difference Moisture Index (NDMI)**  (NIR vs. SWIR) is another useful metric for drought stress as it responds to leaf water content.
    

After mapping NDVI and NDWI, we can reduce the image collection to a single composite per period (e.g., an average or median over the season) to get representative index values that are robust to any remaining clouds or noise:

javascript

CopyEdit

`var seasonComposite = s2.median().select(['NDVI','NDWI']);` 

Now we have two key rasters: NDVI and NDWI over our field.  **NDVI**  will flag areas of low vegetation vigor (potential drought or poor growth) and  **NDWI**  may show water features or relative wetness. You can visualize these in Earth Engine’s map or export them. For example, adding the NDVI layer:

javascript

CopyEdit

`Map.centerObject(region, 15); Map.addLayer(seasonComposite.select('NDVI'), {min:0, max:1, palette:['brown','yellow','green']}, 'NDVI');` 

_Example NDVI map (greener = higher NDVI). High NDVI suggests healthy, well-watered crops, while low NDVI (brownish) may indicate sparse or stressed vegetation._

**Tutorial Reference:**  If you’re new to Earth Engine, check out the official NDVI calculation example[developers.google.com](https://developers.google.com/earth-engine/tutorials/tutorial_api_06#:~:text=%2F%2F%20Compute%20the%20Normalized%20Difference,rename%28%27NDVI)[developers.google.com](https://developers.google.com/earth-engine/tutorials/tutorial_api_06#:~:text=match%20at%20L330%20var%20ndvi,B5%27%2C%20%27B4%27%5D%29.rename%28%27NDVI), which shows how to add an NDVI band and visualize it. Google’s Earth Engine guide on  _“NDVI and image collections”_  demonstrates mapping a function over all images[developers.google.com](https://developers.google.com/earth-engine/tutorials/tutorial_api_06#:~:text=var%20addNDVI%20%3D%20function%28image%29%20,B5%27%2C%20%27B4%27%5D%29.rename%28%27NDVI%27%29%3B%20return%20image.addBands%28ndvi%29%3B). These resources provide a gentle introduction to computing indices and handling image collections.

## Step 2: Integrating Rainfall Data (CHIRPS) for Drought Indicators

Drought stress is often correlated with lack of rainfall. We can enrich our dataset with precipitation information using the  **CHIRPS**  dataset.  **CHIRPS (Climate Hazards InfraRed Precipitation with Stations)**  is a 30+ year global rainfall product at ~5 km (~0.05°) resolution[developers.google.com](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY#:~:text=Climate%20Hazards%20Center%20InfraRed%20Precipitation,analysis%20and%20seasonal%20drought%20monitoring), widely used for drought monitoring. GEE provides CHIRPS as an ImageCollection of daily rainfall[developers.google.com](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY#:~:text=%60%20ee.ImageCollection%28%22UCSB).

**Retrieve CHIRPS in GEE:**  We’ll pull the rainfall over our AOI and time of interest. For example, to get the total rainfall during the same June–August 2021 season:

javascript

CopyEdit

`var chirpsDaily = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                    .filterBounds(region)
                    .filterDate('2021-06-01', '2021-08-31'); var totalRain = chirpsDaily.reduce(ee.Reducer.sum()); // sum of 'precipitation' over time` 

This gives an image (`totalRain`) with the band “precipitation_sum” in millimeters over the summer. Over a 1-ha area, CHIRPS pixels are coarse (5km), so essentially we get one value for the whole field (or we could take the mean if the field intersects multiple pixels). You can also compute other drought indicators, such as rainfall anomalies or consecutive dry days, but keep it simple for now. For instance, the average daily rainfall or number of rainy days could be derived from CHIRPS as well[medium.com](https://medium.com/@linhha53/analyzing-chirps-rainfall-data-using-google-earth-engine-bb4901ca29b7#:~:text=Introduction%3A%20Recently%2C%20I%20had%20the,capabilities%20of%20Google%20Earth%20Engine)[medium.com](https://medium.com/@linhha53/analyzing-chirps-rainfall-data-using-google-earth-engine-bb4901ca29b7#:~:text=STEP%202%3A%20Create%20a%20binary,mask%20for%20rainy%20days).

**Why include rainfall?**  During drought, precipitation is below normal, leading to plant water stress (lower NDVI, possibly lower NDWI/NDMI). By adding a feature for recent rainfall, our model can better distinguish true drought-induced low NDVI from other causes (like harvest or soil issues). In a supervised scenario, rainfall can serve as a proxy for drought conditions to label training data (e.g., periods with <X mm rain in last N days could be marked as “drought” events).

**Efficiency Tip:**  We use a  **small spatial and temporal subset**  for CHIRPS. The data volume for one season over 5 km² is tiny (a few kilobytes), so memory usage stays negligible. Avoid requesting the full global CHIRPS dataset; always filter by date and region to stay within the <500 MB goal.

**Resources:**  To learn more about processing CHIRPS in Earth Engine, see  _Analyzing CHIRPS rainfall data_  on Medium[medium.com](https://medium.com/@linhha53/analyzing-chirps-rainfall-data-using-google-earth-engine-bb4901ca29b7#:~:text=Introduction%3A%20Recently%2C%20I%20had%20the,capabilities%20of%20Google%20Earth%20Engine)[medium.com](https://medium.com/@linhha53/analyzing-chirps-rainfall-data-using-google-earth-engine-bb4901ca29b7#:~:text=,CHG%2FCHIRPS%2FDAILY%27%29%20.filterDate%28start_date%2C%20end_date), which shows how to filter and aggregate rainfall. The Earth Engine Data Catalog also provides the snippet and description for CHIRPS[developers.google.com](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY#:~:text=Climate%20Hazards%20Center%20InfraRed%20Precipitation,analysis%20and%20seasonal%20drought%20monitoring). These can help you adapt precipitation analysis to your needs.

## Step 3: Preparing Training Data (Features and Labels)

With remote sensing features prepared (NDVI, NDWI, and rainfall), we compile a dataset for our ML model. In this stage, aim for  **clarity and minimalism**  – a small table of labeled examples is enough.

**Construct Feature Vectors:**  Each training sample could correspond to a location (or an aggregate region) at a given time. For a simple hackathon demo, you have a few options:

-   **Per-Pixel Samples:**  Use each pixel in the field (or a subset of pixels) as a sample. For each pixel, extract NDVI, NDWI, and total rainfall. This could give on the order of 100 samples for a 1 ha field (10 m pixels), which is manageable. For example, in GEE you might do:
    
    javascript
    
    CopyEdit
    
    `var features = seasonComposite.addBands(totalRain)
                    .sample({region: region, scale: 10, numPixels: 100});` 
    
    This yields a FeatureCollection where each feature has properties  `NDVI`,  `NDWI`, and  `precipitation_sum`. You can convert this to a format for training (export as CSV or get it via  `ee.FeatureCollection.getInfo()`  for small samples).
    
-   **Aggregated Tile as a Sample:**  Alternatively, treat the whole 1-ha tile as one sample with mean or median NDVI/NDWI. This is useful if you have multiple fields to compare. For one field though, it reduces to a single data point (insufficient for ML), so it’s better if you have several tiles (e.g. multiple fields with known stressed vs normal conditions).
    

**Labeling the Data:**  Supervised learning requires a drought stress label. In a hackathon, ground truth might be unavailable, so you can get creative:

-   **Derived Labels:**  Define a heuristic rule to label “drought stress” vs “no stress.” For example, if cumulative rainfall over the period is below a threshold and NDVI is low, label that sample as “drought=1” (stressed). Otherwise label “drought=0”. You might use an NDVI cutoff (say NDVI < 0.3) combined with low rain to auto-label some pixels as drought-stressed. While not perfect, this can generate a pseudo-training set.
    
-   **External Data:**  If any sample data is provided (e.g. known irrigation vs non-irrigated plots, or crop yield loss areas), use those as labels. Even a handful of points drawn in the GEE map (using Geometry drawing tools) can serve as training data[developers.google.com](https://developers.google.com/earth-engine/guides/classification#:~:text=RandomForest%2C%20NaiveBayes%20and%20SVM,general%20workflow%20for%20classification%20is)  (mark some areas in the field as stressed vs healthy based on visual interpretation of imagery).
    

Ensure the labels are balanced (have some of each class) if possible. Since we are keeping the dataset tiny, even  **dozens of samples**  in total can suffice for a demo.

**Finalize the Training Table:**  The outcome should be a table with columns like  `[NDVI, NDWI, Rainfall, Label]`. An example might look like:

NDVI

NDWI

Rain(mm)

DroughtStress (label)

0.45

0.10

50

0 (no drought)

0.25

0.05

10

1 (drought)

...

...

...

...

For such a small table, saving it as a CSV or JSON (a few kilobytes) will be trivial in size. This meets our efficiency goal easily – on the order of  **< 1 MB**.

_Note:_  If doing everything inside Earth Engine, you can keep the data in memory as an  `ee.FeatureCollection`. If moving to Python for ML, use  `Export.table.toDrive`  or  `geemap`  to download the samples. Given the size, you could even manually copy the values.

## Step 4: Training a Simple Machine Learning Model

With features and labels prepared, we train a lightweight model to detect drought stress. We’ll use a  **simple classifier**  like Random Forest (RF) or LightGBM, avoiding deep learning to minimize setup and ensure fast training.

**Choose Your ML Environment:**  You have two main options:

-   _Option A: Train within Earth Engine:_  GEE’s  `Classifier`  package supports CART, Random Forest, SVM, etc., running on Google’s servers[developers.google.com](https://developers.google.com/earth-engine/guides/classification#:~:text=The%20,general%20workflow%20for%20classification%20is). This is convenient for mapping results back onto imagery. For example, you can do:
    
    javascript
    
    CopyEdit
    
    `var trainedRF = ee.Classifier.smileRandomForest(10).train({ features: features, // FeatureCollection from Step 3  classProperty: 'DroughtStress', inputProperties: ['NDVI','NDWI','precipitation_sum']
    }); var classifiedImage = seasonComposite.classify(trainedRF);` 
    
    This trains a Random Forest with 10 trees on our small dataset and classifies the seasonal composite image. Earth Engine handles the heavy lifting (though our dataset is tiny anyway).
    
-   _Option B: Train in Python (e.g. Colab):_  If you prefer scikit-learn or LightGBM, you can export the training data and use those libraries. For instance, using scikit-learn:
    
    python
    
    CopyEdit
    
    `import pandas as pd from sklearn.ensemble import RandomForestClassifier
    
    data = pd.read_csv('drought_training_samples.csv')
    X = data[['NDVI','NDWI','precipitation_sum']]
    y = data['DroughtStress']
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)` 
    
    This should run almost instantly given the tiny dataset (well under 1 MB). You can similarly use LightGBM:
    
    python
    
    CopyEdit
    
    `import lightgbm as lgb
    train_data = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "verbosity": -1}
    lgb_model = lgb.train(params, train_data, num_boost_round=20)` 
    
    Again, training will be very fast. LightGBM is a bit overkill for so few samples, but it’s lightweight and could be handy if you had a larger dataset later.
    

**Interpreting the Model:**  With such simple models, you can inspect feature importances (e.g., RF feature importance) to see which inputs contribute most – likely NDVI (and maybe rainfall) will be top indicators for drought stress. The simplicity helps with interpretability, aligning with the goal of easy-to-understand output.

**Memory/Compute Check:**  Both approaches are extremely light. The Earth Engine training uses cloud compute (no burden on your machine). The Python approach on, say, Google Colab uses perhaps a few MB of RAM. We kept data small (features count in the tens or hundreds), so we are well within the <1 GB RAM and <500 MB data budget.

**Resources:**  If you opt for Earth Engine’s approach, see the  _Supervised Classification_  guide[developers.google.com](https://developers.google.com/earth-engine/guides/classification#:~:text=The%20,general%20workflow%20for%20classification%20is)  which outlines the workflow (collect training data, train classifier, classify image). Google also has a neat example in their tutorials section for crop classification in Colab[colab.research.google.com](https://colab.research.google.com/github/developmentseed/sat-ml-training/blob/main/_notebooks/2020-02-22-Randomforest_cropmapping-with_GEE.ipynb#:~:text=Random%20Forest%20Model%20for%20Crop,mapping%20with%20a%20RandomForest%20Classifier)  –  _“Random Forest Model for Crop Type Mapping”_. While that example is more advanced (and larger scale), the principles are the same. For LightGBM, the official docs or Kaggle tutorials on LightGBM will show you how to quickly train and predict with small datasets.

## Step 5: Producing Drought Stress Maps and Flags

The final step is to apply the model and visualize the results, highlighting areas of potential drought stress.

**If using Earth Engine (Option A):**  You likely already obtained a classified image in Step 4 (`classifiedImage`). This raster might have values 0 (no drought) and 1 (drought) per pixel. You can overlay this on a map as a heatmap or mask:

javascript

CopyEdit

`// Visualization: red for stressed, green for okay  Map.addLayer(classifiedImage, {min:0, max:1, palette:['green','red']}, 'Drought Stress');` 

This will show a  **binary drought stress map**  where red pixels indicate predicted drought stress. In a 1-ha field example, you might only see a handful of pixels, but it demonstrates the concept (e.g., perhaps the edges of the field are red due to weaker irrigation coverage, etc.). For a smoother look, you could compute a “stress probability” if using a model that supports it (e.g., in Random Forest you could output the vote fraction as a pseudo-probability heatmap).

**If using external Python model (Option B):**  After training, you can feed the model new data (e.g., the same features from our season composite) to predict drought stress:

python

CopyEdit

`# Assuming seasonComposite features for each pixel are in dataframe new_X predictions = model.predict(new_X)` 

Then reshape these predictions back to the spatial layout. Since our field is small, one way is to export the NDVI/NDWI image and then re-import the predictions as an image for mapping. But a simpler hack: you could directly threshold NDVI for visualization. For instance, highlight areas with NDVI below a certain value as drought-risk. This isn’t using the full model, but often  **NDVI alone is a decent indicator**  of drought stress (when used at the right time of season). For a hackathon demo, this shortcut can produce a quick map if integrating the ML output is challenging.

That said, to stay true to the model output, you could also upload the CSV of pixel predictions to GEE as a table and display points, though that’s not ideal. A more direct approach is to run the model inside a Jupyter environment that allows mapping (e.g., using  `folium`  or  `geemap`  to display the field and color it by prediction). The specifics may depend on what environment you’re comfortable with.

**Result Interpretation:**  The output is intentionally simple – either a  **heatmap of stress**  (if using probabilities or NDVI values) or a  **binary flag map**. This makes it easy for a farmer or analyst to see “where is the drought stress?” For example, you could produce an image like:

-   A map with the field outlined, green where vegetation is fine and red where drought stress is detected.
    
-   A binary label per field (if each field is one sample, say in a larger analysis) indicating whether it’s under drought stress, which can be presented as a list or table.
    

Such outputs align with our inspiration: real platforms rank fields by issues and highlight problem spots[news.agropages.com](https://news.agropages.com/News/NewsDetail---30223.htm#:~:text=,rank%20issues%20in%20their). Here, our focus is just drought stress, so the map is the key visual.

Finally, you can complement the map with a quick chart or statistic: e.g., “vegetation index dropped 30% below average in stressed areas” or “rainfall was 80% below normal in July”. These context points, while not required, can make your hackathon presentation more compelling.

**Visualization Tools:**  Don’t hesitate to use GEE’s timelapse or charts if relevant – for example, an NDVI time-series chart for the field can show when the decline happened. But given the 2-day limit, a static map is usually enough. If using Python, libraries like matplotlib or plotly can make a colored image from the prediction matrix.

## Leveraging Existing Scripts and Resources

To speed up development, you can  **reuse snippets from public repositories**. The Sentinel Hub  **Custom Scripts**repository is a treasure trove of ready-made formulas for remote sensing indices and algorithms. Below we summarize a few relevant scripts that align with our workflow – you can study these for reference or even adapt parts of them in your code:

-   **NDVI (Normalized Difference Vegetation Index):**  _Sentinel Hub’s NDVI script_  provides the NDVI formula and color visualization tips. NDVI is calculated as (B8 – B4) / (B8 + B4) for Sentinel-2[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi/#:~:text=The%20normalized%20difference%20vegetation%20index%2C,abbreviated%20NDVI%2C%20is%20defined%20as). The script notes NDVI’s range (-1 to 1) and typical values (e.g., >0.5 for dense vegetation)[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi/#:~:text=The%20value%20range%20of%20the,1%5D%20for%20details). This confirms our Earth Engine approach. You can also see how they mask clouds and use color ramps.  _(Link:_  Sentinel Hub NDVI custom script)*
    
-   **NDWI (Normalized Difference Water Index):**  The custom script for NDWI (green vs NIR) highlights water bodies[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/#:~:text=The%20NDWI%20is%20used%20to,was%20proposed%20by%20McFeeters%2C%201996). It notes that NDWI>0.5 often corresponds to water, while vegetation gives smaller values[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/#:~:text=,B04%20%2B%20B02). It also clarifies the distinction from NDMI (NIR vs SWIR)[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/#:~:text=Note%3A%20NDWI%20index%20is%20often,NIR). If your drought analysis needed to detect open water (like pond levels or irrigation water), this script is helpful.  _(Link:_  Sentinel Hub NDWI custom script)*
    
-   **Soil Moisture Estimation (Sentinel-1):**  There’s an advanced script using Sentinel-1 SAR data to estimate surface soil moisture[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-1/soil_moisture_estimation/#:~:text=Script%20estimates%20surface%20soil%20moisture,Permanent). It applies a change detection algorithm (TU Wien model) on a long time series of radar backscatter to infer 0–60% soil moisture, with dry soil in red and wet in blue[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-1/soil_moisture_estimation/#:~:text=Change%20Detection%20model%29,masked%20out%20using%20backscatter%20intensity). Incorporating radar can improve drought detection (since it senses moisture directly, unlike NDVI). However, implementing this in 2 days is challenging. You might not use it during the hackathon, but it’s worth noting for future work. The script’s logic could be replicated in GEE if needed.  _(Link:_  Sentinel Hub Sentinel-1 Soil Moisture script)*
    
-   **Chlorophyll or Nutrient Indices:**  Crop stress can also be related to nutrient deficiency. Sentinel Hub offers scripts for chlorophyll estimation, such as  **Chlorophyll Index – Red Edge (CI<sub>red-edge</sub>)**  and  **Normalized Difference Red Edge (NDRE)**.  **NDRE**  in particular is an index similar to NDVI but using a red-edge band instead of red, making it sensitive to chlorophyll content in leaves[eos.com](https://eos.com/make-an-analysis/ndre/#:~:text=NDRE%20is%20a%20vegetation%20index,The%20NDRE%20formula%20is). It’s defined as (NIR – RedEdge) / (NIR + RedEdge)[eos.com](https://eos.com/make-an-analysis/ndre/#:~:text=represented%20by%20a%20certain%20value,The%20NDRE%20formula%20is). NDRE is useful in later growth stages to detect subtle stress that NDVI might miss[eos.com](https://eos.com/make-an-analysis/ndre/#:~:text=Normalized%20difference%20red%20edge%20index,be%20less%20effective%20to%20use). While our pipeline focused on drought (water stress), you could include NDRE to capture general crop health. Sentinel Hub’s repository provides formulas for NDRE and others (e.g., CCCI – canopy chlorophyll index, various red-edge indices[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/#:~:text=Chlgreen%20Chlorophyll%20Green%20%20id_251,Copernicus%20Browser)[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/#:~:text=CIgreen%20Chlorophyll%20Index%20Green%20,js%20Copernicus%20Browser)). These can be computed in GEE similarly by using the appropriate Sentinel-2 bands (B5, B6, B7 are red-edge bands).
    
-   **Other Indices:**  The custom scripts repository is extensive. Notable mentions for agriculture include  **EVI**(Enhanced Vegetation Index),  **SAVI**  (Soil-Adjusted VI),  **NDMI**  (Moisture Index), and even  **drought indices**  like the  **Normalized Burn Ratio (NBR)**  which can indicate vegetation stress and dry matter. Browse the Sentinel-2 index list[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/#:~:text=NDVI%20Normalized%20Difference%20NIR%2FRed%20Normalized,Normalized%20Difference%20Red%2FGreen%20Redness%20Index)[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/#:~:text=NDRE%20Normalized%20Difference%20NIR%2FRededge%20Normalized,Edge%20id_223.js%20Copernicus%20Browser)  to find formulas. During the hackathon, you probably won’t implement many of these due to time, but knowing they exist means you won’t reinvent the wheel.
    

**Tutorials & Colab Notebooks:**  For each component of this pipeline, there are community tutorials to guide you:

-   _Google Earth Engine Quickstart:_  If you need a quick intro or refresher, the official GEE docs and the  **Earth Engine API Python Notebooks**  (on Google Developers) are great. For instance, the  **NDVI mapping tutorial**[developers.google.com](https://developers.google.com/earth-engine/tutorials/tutorial_api_06#:~:text=%2F%2F%20Compute%20the%20Normalized%20Difference,rename%28%27NDVI)[developers.google.com](https://developers.google.com/earth-engine/tutorials/tutorial_api_06#:~:text=match%20at%20L330%20var%20ndvi,B5%27%2C%20%27B4%27%5D%29.rename%28%27NDVI)  and the  **cloud masking example**[developers.google.com](https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless#:~:text=Engine%20developers,data%20using%20Earth%20Engine)  can be very helpful.
    
-   _Colab Notebooks:_  Google provides ready-to-run Colab notebooks for common tasks. A relevant example is  _“Random Forest for crop classification”_[colab.research.google.com](https://colab.research.google.com/github/developmentseed/sat-ml-training/blob/main/_notebooks/2020-02-22-Randomforest_cropmapping-with_GEE.ipynb#:~:text=Random%20Forest%20Model%20for%20Crop,mapping%20with%20a%20RandomForest%20Classifier)  which shows reading Sentinel-2 in Python, similar to our Step 1 and 4. You can adapt it by changing the features to NDVI/NDWI and focusing on two classes (drought/no drought).
    
-   _Sentinel Hub Playground:_  If GEE is new to some team members, the Sentinel Hub  **EO Browser**  can be a quick way to visualize NDVI or NDWI for your area without coding. You can even plug in the custom scripts (like NDVI or NDWI) into EO Browser’s custom script feature to see results instantly[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi/#:~:text=Evaluate%20and%20Visualize)[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/#:~:text=General%20description%20of%20the%20script). This isn’t part of the pipeline per se, but can help verify that your chosen area and dates show stress signals (e.g., NDVI differences between dates).
    

## Conclusion

In summary, our 2-day hackathon pipeline utilizes  **free satellite imagery (Sentinel-2)**  and  **open climate data (CHIRPS)**in Google Earth Engine to detect drought stress in crops. We focused on a  **tiny spatial extent**  (1 ha) and minimal data to ensure the process is rapid and easily repeatable. The key outputs – NDVI and NDWI maps, rainfall aggregates, and a basic Random Forest classification – are all straightforward to interpret and communicate. Despite its simplicity, this pipeline can flag areas of potential drought stress, fulfilling a core goal inspired by commercial platforms (early warning of water stress[ceres.ai](https://ceres.ai/blog/ceres-ai-and-bayer-climate-fieldview-partner-to-empower-farm-operations-and-financial-stakeholders-with-ai-driven-insights#:~:text=%2A%20Enhanced%20field,drought%20stress%20and%20pest%20outbreaks)).

By following this guide, beginners can learn how each piece fits together: from  **data preprocessing (cloud masking, index calculation)**,  **feature engineering (combining remote sensing with weather)**, to  **model training and mapping results**. Each stage comes with abundant tutorials and even ready-made scripts, so you can stand on the shoulders of existing work rather than starting from scratch. This leaves you more time within the hackathon to experiment and tweak – for example, trying a different index, adjusting the threshold for stress, or validating the output with any available ground truth.

Remember, the emphasis is on  _lightweight and clear_  methods. It’s better to have a simple working NDVI-based alert map than an overly complex model that isn’t finished by demo time. Once the basic pipeline is in place, you can always iterate: perhaps incorporate temporal NDVI changes, or add Sentinel-1 moisture index for improved accuracy, etc. But even as-is, you have a functional prototype for  **crop drought stress detection**  that runs with minimal computing needs. Good luck, and have fun hacking a greener future!

**Sources:**  This guide aggregated best practices and scripts from Google Earth Engine documentation, Sentinel Hub’s custom script repository, and remote sensing literature to ensure each step is grounded in proven methods[developers.google.com](https://developers.google.com/earth-engine/guides/classification#:~:text=The%20,general%20workflow%20for%20classification%20is)[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi/#:~:text=The%20normalized%20difference%20vegetation%20index%2C,abbreviated%20NDVI%2C%20is%20defined%20as)[custom-scripts.sentinel-hub.com](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/#:~:text=The%20NDWI%20is%20used%20to,was%20proposed%20by%20McFeeters%2C%201996)[developers.google.com](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY#:~:text=Climate%20Hazards%20Center%20InfraRed%20Precipitation,analysis%20and%20seasonal%20drought%20monitoring)[eos.com](https://eos.com/make-an-analysis/ndre/#:~:text=NDRE%20is%20a%20vegetation%20index,The%20NDRE%20formula%20is). All code snippets are for demonstration – consult the linked references and tutorials for deeper dives into each topic as needed. Enjoy your hackathon project!
