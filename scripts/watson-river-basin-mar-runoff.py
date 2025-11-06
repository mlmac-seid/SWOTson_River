# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:22:13 2025

@author: mayam
"""

import numpy as np
import os
import xarray as xr
import rasterio
import geopandas as gpd
import rioxarray as rio

# Set working directory
os.chdir('/Users/mayam/OneDrive/Documents/Duke University/ECS 851')

# Open MAR file
mar = xr.open_dataset('MARv3.14.1-5km-daily-ERA5-2024.nc')

# Open Watson River Basin delineation
watson_river_basin = gpd.read_file('/Users/mayam/OneDrive/Documents/Duke University/ECS 851/SWOTson_River/watson-river-basin-delineation/watson_river_basin.shp')

# Convert MAR coordinates from km to meters
mar = mar.assign_coords({"x": mar["x"] * 1000, "y": mar["y"] * 1000,})

# Assign the correct CRS to MAR from its metadata
mar = mar.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
mar = mar.rio.write_crs("EPSG:3413")

# Reproject basin delineation to MAR CRS
watson_river_basin = watson_river_basin.to_crs(mar.rio.crs)

# Clip MAR to the extent of the Watson River Basin
mar_wr = mar.rio.clip(watson_river_basin.geometry, crs=watson_river_basin.crs, drop=True)

# Subset for summer
mar_wr = mar_wr.sel(TIME=slice("2024-05-01", "2024-09-01"))

# Save to a new NetCDF
if os.path.exists('/Users/mayam/OneDrive/Documents/Duke University/ECS 851/SWOTson_River/mar_watson_river_clipped.nc'):
    os.remove('/Users/mayam/OneDrive/Documents/Duke University/ECS 851/SWOTson_River/mar_watson_river_clipped.nc')
mar_wr.to_netcdf('/Users/mayam/OneDrive/Documents/Duke University/ECS 851/SWOTson_River/mar_watson_river_clipped.nc')
