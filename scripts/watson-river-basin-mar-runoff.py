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
import datetime as datetime
from scipy import stats
from scipy.stats import pearsonr

# Set working directory
os.chdir('/Users/mayam/OneDrive/Documents/Duke University/ECS 851')

# Open MAR file
mar = xr.open_dataset('MARv3.14.1-5km-daily-ERA5-2024.nc')

# Open SWOT reach file
swot_df_filtered = pd.read_csv('/Users/mayam/OneDrive/Documents/Duke University/ECS 851/SWOTson_River/data/Watson_filtered_reach_df.csv')

# Open SWOT node 912708000050221 file
swot_node_912708000050221 = pd.read_csv('/Users/mayam/OneDrive/Documents/Duke University/ECS 851/SWOTson_River/data/SWOT_node_912708000050221.csv')

# Open Sentinel-2 effective width file
effective_width = pd.read_csv('/Users/mayam/OneDrive/Documents/Duke University/ECS 851/SWOTson_River/data/effective_width.csv')

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

# Filter SWOT to uppermost reach for comparison
reach_91270800051_mask = swot_df_filtered['reach_id'] ==  91270800051
swot_reach_91270800051 = swot_df_filtered[reach_91270800051_mask]

# Filter SWOT to middle reach for comparison
reach_91270800041_mask = swot_df_filtered['reach_id'] ==  91270800041
swot_reach_91270800041 = swot_df_filtered[reach_91270800041_mask]

# Filter SWOT to lowest reach for comparison
reach_91270800031_mask = swot_df_filtered['reach_id'] ==  91270800031
swot_reach_91270800031 = swot_df_filtered[reach_91270800031_mask]

# Convert SWOT time to datetime array
swot_reach_91270800051['time_str'] = pd.to_datetime(swot_reach_91270800051['time_str'])
swot_node_912708000050221['time_str'] = pd.to_datetime(swot_node_912708000050221['time_str'])
swot_reach_91270800041['time_str'] = pd.to_datetime(swot_reach_91270800041['time_str'])
swot_reach_91270800031['time_str'] = pd.to_datetime(swot_reach_91270800031['time_str'])

# Rename SWOT time to DATE
swot_reach_91270800051 = swot_reach_91270800051.rename(columns={'time_str': 'DATE'})
swot_node_912708000050221 = swot_node_912708000050221.rename(columns={'time_str': 'DATE'})
swot_reach_91270800041 = swot_reach_91270800041.rename(columns={'time_str': 'DATE'})
swot_reach_91270800031 = swot_reach_91270800031.rename(columns={'time_str': 'DATE'})

# Rename effective width date to DATE
effective_width = effective_width.rename(columns={'date': 'DATE'})

# Put MAR time into UTC
mar_runoff_df['DATE'] = mar_runoff_df['DATE'].dt.tz_localize('UTC')

# Put effective width date into UTC
effective_width['DATE'] = pd.to_datetime(effective_width['DATE'])
effective_width['DATE'] = effective_width['DATE'].dt.tz_localize('UTC')
effective_width = effective_width[effective_width['DATE'].dt.year == 2024]

# Merge MAR runoff df to SWOT reach 91270800051 df
merged_swot_reach_91270800051_mar = pd.merge_asof(swot_reach_91270800051, mar_runoff_df, on='DATE')

# Merge MAR runoff df to SWOT reach 91270800041 df
merged_swot_reach_91270800041_mar = pd.merge_asof(swot_reach_91270800041, mar_runoff_df, on='DATE')

# Merge MAR runoff df to SWOT reach 91270800031 df
merged_swot_reach_91270800031_mar = pd.merge_asof(swot_reach_91270800031, mar_runoff_df, on='DATE')

# Merge MAR runoff df to SWOT node 912708000050221 df
merged_swot_node_912708000050221_mar = pd.merge_asof(swot_node_912708000050221, mar_runoff_df, on='DATE')

# Merge MAR runoff df to effective width df
merged_effective_width_mar = pd.merge_asof(effective_width, mar_runoff_df, on='DATE')

# Scatter plot of width vs runoff, colored by quality flag for middle reach:
quality_flags = merged_swot_reach_91270800051_mar['reach_q_b'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(quality_flags)))
color_map = dict(zip(quality_flags, colors))

plt.scatter(
    merged_swot_reach_91270800051_mar['width'],
    merged_swot_reach_91270800051_mar['total_runoff'],
    c=merged_swot_reach_91270800051_mar['reach_q_b'].map(color_map),
    s=50,
    edgecolor='k')

for flag in quality_flags:
    plt.scatter([], [], color=color_map[flag], label=f'{flag}')
plt.legend(title='Quality Flag', bbox_to_anchor=(1, 1))

x = merged_swot_reach_91270800051_mar['width']
y = merged_swot_reach_91270800051_mar['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Width (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and Width')
plt.show()

# Scatter plot of width vs runoff, but only good quality flags
good_flags = [532494, 524290, 8206]
quality_points = merged_swot_reach_91270800051_mar[
    merged_swot_reach_91270800051_mar['reach_q_b'].isin(good_flags)]
    
quality_flags = quality_points['reach_q_b'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(quality_flags)))
color_map = dict(zip(quality_flags, colors))

plt.scatter(
    quality_points['width'],
    quality_points['total_runoff'],
    c=quality_points['reach_q_b'].map(color_map),
    s=50,
    edgecolor='k')

for flag in quality_flags:
    plt.scatter([], [], color=color_map[flag], label=f'{flag}')
plt.legend(title='Quality Flag', bbox_to_anchor=(1, 1))

x = quality_points['width']
y = quality_points['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Width (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and Width')
plt.show()

# Scatter plot of width vs runoff, but only best quality flag
good_flags = [8206]
quality_points = merged_swot_reach_91270800051_mar[
    merged_swot_reach_91270800051_mar['reach_q_b'].isin(good_flags)]
    
quality_flags = quality_points['reach_q_b'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(quality_flags)))
color_map = dict(zip(quality_flags, colors))

plt.scatter(
    quality_points['width'],
    quality_points['total_runoff'],
    c=quality_points['reach_q_b'].map(color_map),
    s=50,
    edgecolor='k')

for flag in quality_flags:
    plt.scatter([], [], color=color_map[flag], label=f'{flag}')
plt.legend(title='Quality Flag', bbox_to_anchor=(1, 1))

x = quality_points['width']
y = quality_points['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Width (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and Width')
plt.show()

# Scatter plot of width vs runoff, colored by quality flag for node 912708000050221
quality_flags = merged_swot_node_912708000050221_mar['node_q_b'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(quality_flags)))
color_map = dict(zip(quality_flags, colors))

plt.scatter(
    merged_swot_node_912708000050221_mar['width'],
    merged_swot_node_912708000050221_mar['total_runoff'],
    c=merged_swot_node_912708000050221_mar['node_q_b'].map(color_map),
    s=50,
    edgecolor='k')

for flag in quality_flags:
    plt.scatter([], [], color=color_map[flag], label=f'{flag}')
plt.legend(title='Quality Flag', bbox_to_anchor=(1, 1))

x = merged_swot_node_912708000050221_mar['width']
y = merged_swot_node_912708000050221_mar['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Width (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and Width')
plt.show()

# Scatter plot of wse vs runoff, colored by quality flag:
quality_flags = merged_swot_reach_91270800051_mar['reach_q_b'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(quality_flags)))
color_map = dict(zip(quality_flags, colors))

plt.scatter(
    merged_swot_reach_91270800051_mar['wse'],
    merged_swot_reach_91270800051_mar['total_runoff'],
    c=merged_swot_reach_91270800051_mar['reach_q_b'].map(color_map),
    s=50,
    edgecolor='k')

for flag in quality_flags:
    plt.scatter([], [], color=color_map[flag], label=f'{flag}')
plt.legend(title='Quality Flag', bbox_to_anchor=(1, 1))

x = merged_swot_reach_91270800051_mar['wse']
y = merged_swot_reach_91270800051_mar['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('WSE (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and WSE')
plt.show()

# Scatter plot of wse vs runoff, but only good quality flags
good_flags = [532494, 524290, 8206]
quality_points = merged_swot_reach_91270800051_mar[
    merged_swot_reach_91270800051_mar['reach_q_b'].isin(good_flags)]
    
quality_flags = quality_points['reach_q_b'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(quality_flags)))
color_map = dict(zip(quality_flags, colors))

plt.scatter(
    quality_points['wse'],
    quality_points['total_runoff'],
    c=quality_points['reach_q_b'].map(color_map),
    s=50,
    edgecolor='k')

for flag in quality_flags:
    plt.scatter([], [], color=color_map[flag], label=f'{flag}')
plt.legend(title='Quality Flag', bbox_to_anchor=(1, 1))

x = quality_points['wse']
y = quality_points['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('WSE (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and WSE')
plt.show()

# Scatter plot of wse vs runoff, but only best quality flag
good_flags = [8206]
quality_points = merged_swot_reach_91270800051_mar[
    merged_swot_reach_91270800051_mar['reach_q_b'].isin(good_flags)]
    
quality_flags = quality_points['reach_q_b'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(quality_flags)))
color_map = dict(zip(quality_flags, colors))

plt.scatter(
    quality_points['wse'],
    quality_points['total_runoff'],
    c=quality_points['reach_q_b'].map(color_map),
    s=50,
    edgecolor='k')

for flag in quality_flags:
    plt.scatter([], [], color=color_map[flag], label=f'{flag}')
plt.legend(title='Quality Flag', bbox_to_anchor=(1, 1))

x = quality_points['wse']
y = quality_points['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('WSE (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and WSE')
plt.show()

# Scatter plot of wse vs runoff, colored by quality flag for node 912708000050221
quality_flags = merged_swot_node_912708000050221_mar['node_q_b'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(quality_flags)))
color_map = dict(zip(quality_flags, colors))

plt.scatter(
    merged_swot_node_912708000050221_mar['wse'],
    merged_swot_node_912708000050221_mar['total_runoff'],
    c=merged_swot_node_912708000050221_mar['node_q_b'].map(color_map),
    s=50,
    edgecolor='k')

for flag in quality_flags:
    plt.scatter([], [], color=color_map[flag], label=f'{flag}')
plt.legend(title='Quality Flag', bbox_to_anchor=(1, 1))

x = merged_swot_node_912708000050221_mar['wse']
y = merged_swot_node_912708000050221_mar['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('WSE (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and WSE')
plt.show()


# Scatter plot of effective width vs runoff
sections = merged_effective_width_mar['section'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(sections)))
color_map = dict(zip(sections, colors))


plt.scatter(
    merged_effective_width_mar['eff_width'],
    merged_effective_width_mar['total_runoff'],
    c=merged_effective_width_mar['section'].map(color_map),
    s=50,
    edgecolor='k')

for sec in sections:
    plt.scatter([], [], color=color_map[sec], label=f'{sec}')
plt.legend(title='Reach', bbox_to_anchor=(1, 1))

x = merged_effective_width_mar['eff_width']
y = merged_effective_width_mar['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Effective Width (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and Width')
plt.show()

# Scatter plot of effective width vs total runoff for upper reach
upper_reach = [1]
upper_reach = merged_effective_width_mar[
    merged_effective_width_mar['section'].isin(upper_reach)]

sections = merged_effective_width_mar['section'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(sections)))
color_map = dict(zip(sections, colors))

plt.scatter(
    upper_reach['eff_width'],
    upper_reach['total_runoff'],
    c=upper_reach['section'].map(color_map),
    s=50,
    edgecolor='k')

x = upper_reach['eff_width']
y = upper_reach['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Effective Width (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and Width')
plt.show()

# Scatter plot of effective width vs total runoff for middle reach
middle_reach = [2]
middle_reach = merged_effective_width_mar[
    merged_effective_width_mar['section'].isin(middle_reach)]

sections = merged_effective_width_mar['section'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(sections)))
color_map = dict(zip(sections, colors))

plt.scatter(
    middle_reach['eff_width'],
    middle_reach['total_runoff'],
    c=upper_reach['section'].map(color_map),
    s=50,
    edgecolor='k')

x = middle_reach['eff_width']
y = middle_reach['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Effective Width (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and Width')
plt.show()

# Scatter plot of effective width vs total runoff for lower reach
lower_reach = [3]
lower_reach = merged_effective_width_mar[
    merged_effective_width_mar['section'].isin(lower_reach)]

sections = merged_effective_width_mar['section'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(sections)))
color_map = dict(zip(sections, colors))

plt.scatter(
    lower_reach['eff_width'],
    lower_reach['total_runoff'],
    c=upper_reach['section'].map(color_map),
    s=50,
    edgecolor='k')

x = lower_reach['eff_width']
y = lower_reach['total_runoff']

x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

r2 = r_value**2
p_formatted = f"{p_value:.4f}"
stats_text = f"$R^2$ = {r2:.3f}\n p = {p_formatted}"
plt.text(
    0.05, 0.95, stats_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.xlabel('Effective Width (m)')
plt.ylabel('Total Runoff (m³)')
plt.title('Relationship Between Watson River Total Runoff and Width')
plt.show()

# Effective width and SWOT width timeseries for upper reach
plt.figure(figsize=(12, 10))
upper_reach = [1]
upper_reach = merged_effective_width_mar[
    merged_effective_width_mar['section'].isin(upper_reach)]
plt.scatter(
    upper_reach['DATE'],
    upper_reach['eff_width'],
    label='Effective Width (Sentinel-2)',
    color='steelblue',
    s=50,        # point size
    edgecolor='k',
    alpha=0.8)
plt.scatter(
    merged_swot_reach_91270800051_mar['DATE'],
    merged_swot_reach_91270800051_mar['width'],
    label='Width (SWOT)',
    color='darkorange',
    s=50,
    edgecolor='k',
    alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Width (m)')
plt.title('Sentinel-2 Effective Width and SWOT Width (Upper Reach)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Effective width and SWOT width timeseries for middle reach
plt.figure(figsize=(12, 10))
middle_reach = [2]
middle_reach = merged_effective_width_mar[
    merged_effective_width_mar['section'].isin(middle_reach)]
plt.scatter(
    middle_reach['DATE'],
    middle_reach['eff_width'],
    label='Effective Width (Sentinel-2)',
    color='steelblue',
    s=50,        # point size
    edgecolor='k',
    alpha=0.8)
plt.scatter(
    merged_swot_reach_91270800041_mar['DATE'],
    merged_swot_reach_91270800041_mar['width'],
    label='Width (SWOT)',
    color='darkorange',
    s=50,
    edgecolor='k',
    alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Width (m)')
plt.title('Sentinel-2 Effective Width and SWOT Width (Middle Reach)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Effective width and SWOT width timeseries for lower reach
plt.figure(figsize=(12, 10))
lower_reach = [3]
lower_reach = merged_effective_width_mar[
    merged_effective_width_mar['section'].isin(lower_reach)]
plt.scatter(
    lower_reach['DATE'],
    lower_reach['eff_width'],
    label='Effective Width (Sentinel-2)',
    color='steelblue',
    s=50,        # point size
    edgecolor='k',
    alpha=0.8)
plt.scatter(
    merged_swot_reach_91270800031_mar['DATE'],
    merged_swot_reach_91270800031_mar['width'],
    label='Width (SWOT)',
    color='darkorange',
    s=50,
    edgecolor='k',
    alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Width (m)')
plt.title('Sentinel-2 Effective Width and SWOT Width (Lower Reach)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Pearson correlation between runoff and effective width in merged_effective_width_mar for reach 1
upper_reach = [1]
upper_reach = merged_effective_width_mar[
    merged_effective_width_mar['section'].isin(upper_reach)]
upper_reach = upper_reach[['eff_width', 'total_runoff']].dropna()
corr_width_upper, p_val_width_upper = pearsonr(upper_reach['eff_width'], upper_reach['total_runoff'])

# Pearson correlation between runoff and effective width in merged_effective_width_mar for reach 2
middle_reach = [2]
middle_reach = merged_effective_width_mar[
    merged_effective_width_mar['section'].isin(middle_reach)]
middle_reach = middle_reach[['eff_width', 'total_runoff']].dropna()
corr_width_middle, p_val_width_middle = pearsonr(middle_reach['eff_width'], middle_reach['total_runoff'])

# Pearson correlation between runoff and effective width in merged_effective_width_mar for reach 3
lower_reach = [3]
lower_reach = merged_effective_width_mar[
    merged_effective_width_mar['section'].isin(lower_reach)]
lower_reach = lower_reach[['eff_width', 'total_runoff']].dropna()
corr_width_lower, p_val_width_lower = pearsonr(lower_reach['eff_width'], lower_reach['total_runoff'])

# Pearson correlation between runoff and width from swot for 51 reach
reach_51 = merged_swot_reach_91270800051_mar[['width', 'total_runoff']].dropna()
corr_width_51, p_val_width_51 = pearsonr(reach_51['width'], reach_51['total_runoff'])

# Pearson correlation between runoff and width from swot for 41 reach
reach_41 = merged_swot_reach_91270800041_mar[['width', 'total_runoff']].dropna()
corr_width_41, p_val_width_41 = pearsonr(reach_41['width'], reach_41['total_runoff'])

# Pearson correlation between runoff and width from swot for 31 reach
reach_31 = merged_swot_reach_91270800031_mar[['width', 'total_runoff']].dropna()
corr_width_31, p_val_width_31 = pearsonr(reach_31['width'], reach_31['total_runoff'])

# Pearson correlation between runoff and wse from swot for 51 reach
reach_51_wse = merged_swot_reach_91270800051_mar[['wse', 'total_runoff']].dropna()
corr_wse_51, p_val_wse_51 = pearsonr(reach_51_wse['wse'], reach_51_wse['total_runoff'])

# Pearson correlation between runoff and wse from swot for 41 reach
reach_41_wse = merged_swot_reach_91270800041_mar[['wse', 'total_runoff']].dropna()
corr_wse_41, p_val_wse_41 = pearsonr(reach_41_wse['wse'], reach_41_wse['total_runoff'])

# Pearson correlation between runoff and wse from swot for 31 reach
reach_31_wse = merged_swot_reach_91270800031_mar[['wse', 'total_runoff']].dropna()
corr_wse_31, p_val_wse_31 = pearsonr(reach_31_wse['wse'], reach_31_wse['total_runoff'])

# Pearson correlation between runoff and wse from swot for node
node_21 = merged_swot_node_912708000050221_mar[['width', 'total_runoff']].dropna()
corr_width_node, p_val_width_node = pearsonr(node_21['width'], node_21['total_runoff'])

# Pearson correlation between runoff and width from swot for node
node_21_wse = merged_swot_node_912708000050221_mar[['wse', 'total_runoff']].dropna()
corr_wse_node, p_val_wse_node = pearsonr(node_21_wse['wse'], node_21_wse['total_runoff'])