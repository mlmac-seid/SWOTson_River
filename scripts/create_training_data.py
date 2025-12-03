# -*- coding: utf-8 -*-
"""
created on: 2025-11-17
@author:    Jasper Heuer
use:        create training and testing data
"""

# import packages =============================================================

import os
import glob
import random
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from shapely.ops import unary_union
from shapely.geometry import Point

# set working directory =======================================================

base_path = "C:/Jasper/PhD/Classes/ECS_851_Geospatial_Data_Science/Project"
os.chdir(base_path)

# define functions ============================================================

def random_points(polygon, n_points, rng=None):
    """
    function to sample random points within a polygon
    """
    rng = rng
    
    x_min, y_min, x_max, y_max = polygon.bounds
    points = []
    
    while len(points) < n_points:
        point = Point(random.uniform(x_min, x_max), 
                      random.uniform(y_min, y_max))
        if polygon.contains(point):
            points.append(point)
            
    return points

# read shapefiles =============================================================

files = glob.glob("./data/shapefiles/*.shp")
lc_classes = [Path(p).stem for p in files]

sampled_points = []

# set random state:
rng42 = np.random.default_rng(42)

for lc_class, shapefile in zip(lc_classes, files):
    gdf = gpd.read_file(shapefile)
    
    # union polygons:
    union = unary_union(gdf.geometry)
    
    # sample random points:
    points = random_points(union, n_points=300, rng=rng42)
    
    for point in points:
        sampled_points.append({
            "x": point.x,
            "y": point.y,
            "landcover": lc_class})
        
# turn to DataFrame and export:
df = pd.DataFrame(sampled_points)

# add landcover as integer column:
lc_dict = {"bedrock_vegetation": 1,
           "lakes": 2,
           "river": 3,
           "river_bank": 4,
           "snow": 5}

df["int_lc"] = df["landcover"].map(lc_dict)

df.to_csv("./data/training_data/sampled_points.csv")

# get raster values at points =================================================

gdf = gpd.GeoDataFrame(df, geometry=[Point(x,y) for x,y in zip(df.x, df.y)], 
                       crs="EPSG:3413")

# get point coordinates:
coords = list(zip(gdf.geometry.x, gdf.geometry.y))

# sample from DEM:
with rasterio.open("./data/training_data/dem_mosaic.tif") as dem_src:
    dem_samples = np.array(list(dem_src.sample(coords)))
    
# sample from NDWI:
with rasterio.open("./data/training_data/20230712_NDWI.tif") as ndwi_ds:
    ndwi_samples = np.array(list(ndwi_ds.sample(coords)))
    
# sample from NDVI:
with rasterio.open("./data/training_data/20230712_NDVI.tif") as ndvi_ds:
    ndvi_samples = np.array(list(ndvi_ds.sample(coords)))
    
# sample from Sentinel-2 image:
with rasterio.open("./data/training_data/20230712T150951_20230712T151003_T22WEV.tif") as img_src:
    img_samples = np.array(list(img_src.sample(coords)))
    
# add data to GeoDataFrame ====================================================

bands = ["blue", "green", "red", "nir"]

for i, band in enumerate(bands):
    gdf[band] = img_samples[:, i] # add image data

# add DEM, NDWI, NDVI:
gdf["dem"] = dem_samples
gdf["ndwi"] = ndwi_samples 
gdf["ndvi"] = ndvi_samples

final_df = gdf.drop(columns="geometry") # drop unnecessary column
final_df = final_df.dropna()

# export as CSV:
final_df.to_csv("./data/training_data/train_dataset.csv")
