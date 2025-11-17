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
import contextily as ctx
import pyproj

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
mar_wr = mar_wr.sel(TIME=slice("2024-06-01", "2024-09-01"))

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

mean_runoff_volume = runoff_volume.mean(dim="TIME")
fig, ax = plt.subplots(figsize=(10, 8))
mean_runoff_volume.plot.imshow(
    ax=ax, 
    cmap='Blues', 
    cbar_kwargs={'label': 'Mean Runoff Volume (m³)'}
)
watson_river_basin.boundary.plot(ax=ax, color='black', linewidth=1)
plt.title("Mean Runoff Volume with Basin Outline (Summer 2024)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()

# Plot Watson River catchment on satellite imagery
fig, ax = plt.subplots(figsize=(8, 8))
# Reproject catchment
watson_river_basin_outline = watson_river_basin.to_crs(epsg=3857)
# Plot basin boundary
watson_river_basin_outline.boundary.plot(ax=ax, color='blue', linewidth=2)
# Get bounds + padding to zoom out
xmin, ymin, xmax, ymax = watson_river_basin_outline.total_bounds
pad_x = (xmax - xmin) * 1.5
pad_y = (ymax - ymin) * 2
lims = [xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y]
ax.set_xlim(lims[0], lims[1])
ax.set_ylim(lims[2], lims[3])
# Add satellite basemap
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=8)
# Transform from Web Mercator to WGS84
transformer = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)
# Get current tick locations (in meters)
xticks = ax.get_xticks()
yticks = ax.get_yticks()
# Convert to lon/lat
lon_labels, _ = transformer.transform(xticks, [ax.get_ylim()[0]] * len(xticks))
_, lat_labels = transformer.transform([ax.get_xlim()[0]] * len(yticks), yticks)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{lon:.2f}°" for lon in lon_labels])
ax.set_yticks(yticks)
ax.set_yticklabels([f"{lat:.2f}°" for lat in lat_labels])
ax.set_title("Watson River Ice Sheet Catchment", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.show()
