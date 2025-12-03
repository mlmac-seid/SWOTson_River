# -*- coding: utf-8 -*-
"""
created on: 2025-11-20
@author:    Jasper Heuer
use:        calculate river area per section
"""

# import packages =============================================================

import os
import glob
import fiona
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from datetime import datetime
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from rasterio.mask import mask

# set working directory =======================================================

base_path = "C:/Jasper/PhD/Classes/ECS_851_Geospatial_Data_Science/Project"
os.chdir(base_path)

# read river section points ===================================================

gdf = gpd.read_file("./data/river_sections/section_coords.shp")
lons = gdf.geometry.x

with fiona.open("./data/masks/river_mask.shp") as shapefile:
    river_mask = [feature["geometry"] for feature in shapefile]

# read landcover rasters ======================================================

lc_files = glob.glob("./data/landcover/good/*.tif")

records = []

for file in lc_files:
    date = file.split("good")[1][1:9]
    dt_date = datetime.strptime(date, "%Y%m%d")

    with rasterio.open(file) as ds:
        out_img, out_trans = mask(ds, river_mask, invert=False)
    
        for i in range(len(lons)-1):
            east = lons[i]
            west = lons[i+1]
            
            if i == 0:
                length = 8172.08
            elif i == 1:
                length = 8382.11
            else:
                length = 8367.81
            
            bounds = (west, ds.bounds.bottom,
                      east, ds.bounds.top)
        
            win = from_bounds(*bounds, transform=out_trans)
            
            row_start = int(win.row_off)
            row_stop  = row_start + int(win.height)
            col_start = int(win.col_off)
            col_stop  = col_start + int(win.width)
            
            lc_slice = out_img[0, row_start:row_stop, col_start:col_stop]
            
            plt.imshow(lc_slice)
            plt.show()
            
            count = np.count_nonzero(lc_slice == 3)
            
            # append data:
            records.append({"date": dt_date,
                            "section": i+1,
                            "area": count * 100,
                            "eff_width": count * 100 / length})  
    ds.close()

# create DataFrame:
df = pd.DataFrame.from_dict(records)
df.to_csv("./data/output/effective_width.csv")

# split by years:
df_2023 = df.iloc[:27, :]
df_2024 = df.iloc[27:, :]

# plot river areas ============================================================

s1_length = 8172.08
s2_length = 8382.11
s3_length = 8367.81

# plot 2023:
plt.plot(df_2023[df_2023["section"] == 1]["date"], 
         df_2023[df_2023["section"] == 1]["eff_width"], 
         label="East")
plt.plot(df_2023[df_2023["section"] == 2]["date"], 
         df_2023[df_2023["section"] == 2]["eff_width"], 
         label="Center")
plt.plot(df_2023[df_2023["section"] == 3]["date"], 
         df_2023[df_2023["section"] == 3]["eff_width"], 
         label="West")
plt.title("Effective width per section (2023)")
plt.ylabel("Effective width in m")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# plot 2024:
plt.plot(df_2024[df_2024["section"] == 1]["date"], 
         df_2024[df_2024["section"] == 1]["eff_width"], 
         label="East")
plt.plot(df_2024[df_2024["section"] == 2]["date"], 
         df_2024[df_2024["section"] == 2]["eff_width"], 
         label="Center")
plt.plot(df_2024[df_2024["section"] == 3]["date"], 
         df_2024[df_2024["section"] == 3]["eff_width"], 
         label="West")
plt.title("Effective width per section (2024)")
plt.ylabel("Effective width in m")
plt.xticks(rotation=45)
plt.legend()
plt.show()