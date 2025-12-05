# -*- coding: utf-8 -*-
"""
created on: 2025-11-14
@author:    Jasper Heuer
use:        calculate NDWI for Watson River Sentinel-2 imagery
"""

# import packages =============================================================

import os
import glob
import rasterio
import matplotlib.pyplot as plt

# set working directory =======================================================

base_path = "C:/Jasper/PhD/Classes/ECS_851_Geospatial_Data_Science/Project"
os.chdir(base_path)

# read data ===================================================================

files = glob.glob("./data/sentinel/input/*.tif", recursive=True)

for file in files: 
    date = file.split("input")[1][1:9]
    
    # open file and read bands:
    ds = rasterio.open(file) 
    green = ds.read(1)
    red = ds.read(2)
    nir = ds.read(3)
    
    # calculate NDWI:
    ndwi = (green - nir) / (green + nir)
    ndvi = (nir - red) / (nir + red)
    
    with rasterio.open(
        f"./data/sentinel/ndwi/{date}_NDWI.tif",
        mode="w",
        driver="GTiff",
        height=green.shape[0],
        width=green.shape[1],
        count=1,
        dtype=green.dtype,
        crs=ds.crs,
        transform=ds.transform
        ) as dst:
            dst.write(ndwi, 1)   
            
    with rasterio.open(
        f"./data/sentinel/ndwi/{date}_NDVI.tif",
        mode="w",
        driver="GTiff",
        height=green.shape[0],
        width=green.shape[1],
        count=1,
        dtype=green.dtype,
        crs=ds.crs,
        transform=ds.transform
        ) as dst:
            dst.write(ndvi, 1)
