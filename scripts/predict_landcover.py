# -*- coding: utf-8 -*-
"""
created on: 2025-11-19
@author:    Jasper Heuer
use:        predict land cover of Watson River ROI
"""

# import packages =============================================================

import os
import glob
import pickle
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

# set working directory =======================================================

base_path = "C:/Jasper/PhD/Classes/ECS_851_Geospatial_Data_Science/Project"
os.chdir(base_path)

# load model ==================================================================

with open("./data/model/RF_IMG_DEM_NDWI_NDVI.pkl","rb") as f:
    clf = pickle.load(f)
    
# score model =================================================================

df = pd.read_csv("./data/testing_data/test_dataset.csv")

X_test = df.drop(columns=["Unnamed: 0", "x", "y", "landcover", "int_lc"])
y_test = df["int_lc"]

results = clf.score(X_test, y_test)

print(f"Model score: {results * 100:.2f}%")

# predict landcover over ROI ==================================================

# get file lists:
img_files = glob.glob("./data/sentinel/input/*.tif")
ndwi_files = glob.glob("./data/sentinel/indices/*NDWI.tif")
ndvi_files = glob.glob("./data/sentinel/indices/*NDVI.tif")

# read DEM data:
dem_ds = rasterio.open("./data/training_data/dem_mosaic.tif")
dem = dem_ds.read()  

# empty list:
records = []

# run predcition loop:    
for img_file, ndwi_file, ndvi_file in zip(img_files, ndwi_files, ndvi_files):
    # read data:
    dem_ds = rasterio.open("./data/training_data/dem_mosaic.tif")
    dem = dem_ds.read()  
    
    img_ds = rasterio.open(img_file)
    ndwi_ds = rasterio.open(ndwi_file)
    ndvi_ds = rasterio.open(ndvi_file)    

    img = img_ds.read()
    ndwi = ndwi_ds.read()
    ndvi = ndvi_ds.read() 
       
    # build stack and flatten it:
    stack = np.dstack([img[0], img[1], img[2], img[3], dem[0], ndwi[0], ndvi[0]])
    flat = stack.reshape(-1, stack.shape[-1])
    
    # get valid pixels mask:
    valid = ~np.any(np.isnan(flat), axis=1)
    
    # create array of zeros and fill with valid pixel predictions
    out_flat = np.zeros(flat.shape[0], dtype=np.int32)
    out_flat[valid] = clf.predict(flat[valid])
    
    # reshape to prediction map:
    pred_map = out_flat.reshape(dem_ds.shape)
    
    # get date:
    date = ndwi_file.split("NDWI.tif")[0][-9:-1]
    dt_date = datetime.strptime(date, "%Y%m%d")
    
    # get water area:
    pixel_counts = np.unique(pred_map, return_counts=True)
    river_area = pixel_counts[1][2] * 100 # in m²
    
    # add to records:
    records.append({"date": dt_date,
                    "river_area": river_area})

    # display map:
    plt.imshow(pred_map)
    plt.title(f"Landcover on {dt_date}")
    plt.show()

    # write to disk:
    with rasterio.open(
        f"./data/landcover/{date}_lc.tif",
        mode="w",
        driver="GTiff",
        height=dem_ds.shape[0],
        width=dem_ds.shape[1],
        count=1,
        dtype=dem.dtype,
        crs=dem_ds.crs,
        transform=dem_ds.transform
        ) as dst:
            dst.write(pred_map, 1)
            
# create DataFrame:
df = pd.DataFrame.from_dict(records)
df.to_csv("./data/output/river_area.csv")
