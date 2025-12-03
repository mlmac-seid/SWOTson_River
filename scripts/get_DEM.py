# -*- coding: utf-8 -*-
"""
created on: 2025-10-28
@author:    Jasper Heuer
use:        find and download ArcticDEM strips through API
"""

# import packages =============================================================

import os
import rasterio
import rioxarray
import numpy as np
import pdemtools as pdt
import geopandas as gpd
import matplotlib.pyplot as plt

# set working directory =======================================================

base_path = "C:/Jasper/PhD/Classes/ECS_851_Geospatial_Data_Science/Project"
os.chdir(base_path)

# define function =============================================================

# define download function:
def download_scene(gdf_row, dem_id, bounds, output_directory):
    """
    function that automatically downloads DEM strip if it is not found in the 
    output directory
    """
    
    # define output path:
    dem_name = dem_id.split("_")[3]
    out_fpath = os.path.join(output_directory, f'DEM_strip_{dem_name}.tif')

    # check whether DEM already exists locally:
    if not os.path.exists(out_fpath):
        dem = pdt.load.from_search(gdf_row, bounds=bounds, bitmask=True)
        dem.rio.to_raster(out_fpath, compress='ZSTD', predictor=3, zlevel=1)
        return dem
    
    else:
        return pdt.load.from_fpath(out_fpath, bounds=bounds)
    
# import ArcticDEM mosaic =====================================================

# get bounds from mask:
gdf = gpd.read_file("./data/masks/mask.shp")
bounds = tuple(gdf.total_bounds.astype(int))
# bounds = (-185100, -2321200, -182000, -2319000) # xmin, ymin, xmax, ymax

# search for mosaic
dem = pdt.load.mosaic(
    dataset='arcticdem',
    resolution=10, # options are 2, 10, and 32 meters
    bounds=bounds,       
    version='v4.1') # defaults to most recent version if not specified

# calculate terrain features:
variable_list = [
    "slope",
    "aspect",
    "hillshade",
    "plan_curvature",
    "horizontal_curvature",
    "vertical_curvature",
    "horizontal_excess",
    "vertical_excess",
    "mean_curvature",
    "gaussian_curvature",
    "unsphericity_curvature",
    "minimal_curvature",
    "maximal_curvature"]

terrain = dem.pdt.terrain(variable_list, hillshade_z_factor=2, 
                          hillshade_multidirectional=True)

# plot DEM tile:
fig, ax = plt.subplots(figsize=(7, 5))

dem.plot.imshow(ax=ax, cmap='gist_earth', vmin=np.min(dem), vmax=np.max(dem))

ax.set_title("DEM mosaic tile")
plt.show()

# plot hillshade:
fig, ax = plt.subplots(figsize=(7, 5))

dem.plot.imshow(ax=ax, cmap='Greys_r', vmin=np.min(dem), vmax=np.max(dem))

terrain.hillshade.plot.imshow(ax=ax, 
                              cmap='Greys_r', 
                              alpha=0.7, 
                              add_colorbar=False)

ax.set_title('Hillshade of DEM mosaic')
plt.show()

# export DEM as GeoTIFF:
dem.rio.to_raster("./data/dem/input/DEM_mosaic.tif", 
                  compress='ZSTD', 
                  predictor=3, # not sure if this is needed/useful?
                  zlevel=1) # compression level

# search for ArcticDEM strips =================================================

# search for strips:
gdf = pdt.search(
    dataset="arcticdem",
    bounds = bounds,
    # one fixed time period:
    # dates = '20240101/20241231',
    # going for seasons:
    # months = [9, 10, 11],
    years = [2020, 2021, 2022, 2023, 2024],
    baseline_max_hours = 24, # max time difference between images for stereo
    sensors=['WV03', 'WV02', 'WV01'], # list of satellite (+ "GE01")
    accuracy=2, # height accuracy in meters - can also take a min/max interval
    min_aoi_frac = 0.5, # area fraction covered by strip
)

print(f'{len(gdf)} strips found')

print(gdf.pdt_datetime1.dt.date.values)

# display hillshade to assess quality:
for i in range(len(gdf)):    
    preview = pdt.load.preview(gdf.iloc[[i]], bounds)

    fig, ax = plt.subplots(layout='constrained')
    ax.set_aspect('equal')
    preview.plot.imshow(cmap='Greys_r', add_colorbar=False)

    gdf.iloc[[i]].plot(ax=ax, fc='none', ec='tab:red')

    ax.set_title(f"{gdf.iloc[[i]].pdt_datetime1.dt.date.values[0]} Index: {i}")
    plt.show()


    
# download sample DEM strip:
output_directory = "./data/GEE_sample_folder/dem/input/"
    
for i in range(len(gdf)): 
    date = gdf.iloc[[i]].pdt_datetime1.dt.date.values[0]
    dem_id = gdf.iloc[[i]].pdt_id.values[0]
    
    # remove unusable DEMs:
    if i in [6]:
        print(f"Skipping DEM strip {i}")
        continue
    # export DEM strips:
    else:
        dem_strip = download_scene(gdf.iloc[i], dem_id, bounds, 
                                   output_directory)
