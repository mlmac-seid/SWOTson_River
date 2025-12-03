# -*- coding: utf-8 -*-
"""
created on: 2025-10-30
@author:    Jasper Heuer (based on Gyula Mate Kovács)
use:        collect and cloud-mask Sentinel-2 imagery
"""

# import packages =============================================================

import os
import ee
import time
import geopandas as gpd

from datetime import datetime

# set working directory =======================================================

base_path = "C:/Jasper/PhD/Classes/ECS_851_Geospatial_Data_Science/Project"
os.chdir(base_path)

# initalize API and read masks ================================================

ee.Authenticate() # ideally only need to do this once
ee.Initialize(project="phd-project-477919")

# import masks and convert to EarthEngine objects:
river_gpd = gpd.read_file("./data/masks/river_mask.shp")
river_gpd = river_gpd.to_crs(epsg=4326)
river_geojson = river_gpd.geometry[0].__geo_interface__
river_mask = ee.Geometry(river_geojson)

roi_gpd = gpd.read_file("./data/masks/mask.shp")
roi_gpd = roi_gpd.to_crs(epsg=4326)
roi_geojson = roi_gpd.geometry[0].__geo_interface__
roi_mask = ee.Geometry(roi_geojson)

# get Sentinel-2 collection ===================================================

dataset = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate("2024-08-01", "2024-08-31")
           .filterBounds(river_mask))

print("Number of images (unfiltered):", dataset.size().getInfo())

# define functions ============================================================

def create_mask(image):
    """
    function to create a snow, cloud, and cloud shadow mask based on pixel
    classification band
    """
    qa = image.select("SCL") # select pixel classification band

    # define masks:
    snow_mask = qa.eq(11).rename("snowmask")
    cloud_mask = (qa.eq(7).Or(qa.eq(8)).Or(qa.eq(9)).Or(qa.eq(10))).rename("cloudmask")
    cloud_shadow_mask = qa.eq(3).rename("shadowmask")

    # add masks as new bands:
    return image.addBands([snow_mask, cloud_mask, cloud_shadow_mask]).clip(roi_mask)

def apply_scale_factors(image):
    """
    function to apply recommended scaling factors to Sentinel-2 bands
    """
    optical_bands = image.select(["B2", "B3", "B4", "B8"]).divide(10000).toFloat()
    
    return image.addBands(optical_bands, overwrite=True).clip(roi_mask)

def compute_ratio(hist_dict):
    """
    function to compute the ratio between cloudy and non-cloudy pixels
    """
    hist = ee.Array(hist_dict.get("histogram"))
    means = ee.Array(hist_dict.get("bucketMeans"))

    size = hist.length().get([0])

    # handle all clear or all cloudy conditions:
    ratio = ee.Algorithms.If(
        # if all cloudy = 1, if all clear = 0:
        ee.Number(size).eq(1),
        ee.Algorithms.If(
            ee.Number(means.get([0])).eq(0), 0, 1),
        # else compute cloud ratio:
        ee.Number(hist.get([1])).divide(ee.Number(hist.get([0])).add(ee.Number(hist.get([1])))))
    
    return ee.Number(ratio)

def cloud_ratio(image):
    """
    function to retrieve the histogram of the reclassification to determine 
    """
    # get cloud pixel histogram:
    count = image.select("cloudmask").reduceRegion(
        reducer=ee.Reducer.histogram(),
        geometry=river_mask,
        scale=10,
        maxPixels=1e10)

    # extract histogram info:
    histogram = ee.Dictionary(count.get("cloudmask"))

    ratio = ee.Algorithms.If(histogram, compute_ratio(histogram), None)
    
    return image.set("CLOUD_RATIO", ratio)

def monitor_tasks(tasks, interval=15):
    """
    function to monitor the export status of the imagery and display any error 
    messages that might be raised
    """
    # get start time for task:
    task_start_times = {t.id: time.time() for t in tasks}

    all_done = False # progress flag
    
    while not all_done:
        
        all_done = True
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking status...")
        print(f"{'Description':35} | {'State':10} | {'Elapsed (s)':>10} | Error")
        print("=" * 80)

        for t in tasks:
            status = t.status() # get status
            state = status['state']
            error = status.get('error_message', '')
            elapsed = int(time.time() - task_start_times[t.id])

            print(f"{status['description'][:35]:35} | {state:10} | {elapsed:10} | {error}")

            if state in ['READY', 'RUNNING']:
                all_done = False  

        if not all_done:
            time.sleep(interval)

    print("Export complete!")

# apply masking and scaling ===================================================

dataset = dataset.map(apply_scale_factors).map(create_mask)

# compute cloud ratio =========================================================

n_img = dataset.size().getInfo()
image_list = dataset.toList(n_img)
image_list2 = ee.List([])

for i in range(n_img):
    image_i = ee.Image(image_list.get(i))
    with_ratio = cloud_ratio(image_i)
    image_list2 = image_list2.add(with_ratio)

cloudratio_dataset = ee.ImageCollection.fromImages(image_list2)

# filter by cloud ratio:
filtered_dataset = cloudratio_dataset.filter(ee.Filter.lte("CLOUD_RATIO", 0.2))
print("Number of images (filtered):", filtered_dataset.size().getInfo())

# export imagery ==============================================================

# list of image IDs:
id_list = filtered_dataset.aggregate_array("system:index").getInfo()

# create empty list of tasks:
tasks = []

# start tasks:
for img_id in id_list:
    image = filtered_dataset.filter(ee.Filter.eq("system:index", img_id)).first()
    mask = image.select("cloudmask").lt(1)
    masked_img = image.updateMask(mask)

    # export statement:
    task = ee.batch.Export.image.toDrive(
        image=masked_img.select(["B2", "B3", "B4", "B8"]),
        description=img_id,
        folder="Watson_River",  # change output folder if needed
        region=roi_mask,
        scale=10,
        crs="EPSG:3413",
        maxPixels=1e13,
        formatOptions={"cloudOptimized": True})
    
    # start task and append to list of tasks to be monitored:
    task.start()
    tasks.append(task)

# monitor progress:
monitor_tasks(tasks, interval=60)
