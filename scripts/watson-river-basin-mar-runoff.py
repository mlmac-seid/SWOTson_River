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
import pandas as pd
import matplotlib.pyplot as plt

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

# Get runoff for Watson River catchment
runoff = mar_wr['RU']
# Use Sector 0 for runoff
runoff = runoff.isel(SECTOR1_1=0)

# Compute area of each grid cell
dx = np.abs(np.diff(mar_wr.x)[0])
dy = np.abs(np.diff(mar_wr.y)[0])
cell_area = dx * dy

# Convert runoff from mmWE to m^3
runoff_m = runoff / 1000.0
# Calculate runoff volume per grid cell
runoff_volume = runoff_m * cell_area

# Sum runoff volume across all grid cells in the catchment
total_runoff = runoff_volume.sum(dim=["y","x"])

# Make a dataframe of dates and total runoff
mar_runoff_df = pd.DataFrame({'DATE' : pd.to_datetime(mar_wr.TIME.values),
                              'total_runoff': total_runoff.values})

# Define output directory
output_dir = '/Users/mayam/OneDrive/Documents/Duke University/ECS 851/SWOTson_River'
# Define output file name with path
ru_file_name = os.path.join(output_dir, "watson_river_runoff.csv")
# Delete old csv file if it exists
if os.path.exists(ru_file_name):
    os.remove(ru_file_name)
# Save runoff to csv
mar_runoff_df.to_csv(ru_file_name, index=False)

# Plot daily total runoff for the catchment
plt.figure(figsize=(20, 6))
plt.plot(mar_runoff_df['DATE'], mar_runoff_df['total_runoff'])
plt.xlabel('Date')
plt.ylabel('Total Runoff (m^3)')
plt.title('Total Runoff for Watson River Catchment for Summer 2024')
plt.show()
