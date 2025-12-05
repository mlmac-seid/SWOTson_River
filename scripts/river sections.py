# -*- coding: utf-8 -*-
"""
created on: 2025-11-20
@author:    Jasper Heuer
use:        create shapefiles outlining river sections
"""

# import packages =============================================================

import os
import geopandas as gpd

from pyproj import Transformer
from shapely.geometry import Point

# set working directory =======================================================

base_path = "C:/Jasper/PhD/Classes/ECS_851_Geospatial_Data_Science/Project"
os.chdir(base_path)

# transform point coordinates =================================================

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)

# define latitude and longitude:
lons = [-50.18852805, -50.34305383, -50.50901778, -50.65789905]
lats = [67.06278316, 67.06288917, 67.03620072, 67.01855919]

# get easting and northing:
xs, ys = transformer.transform(lons, lats)

points = [Point(x, y) for x, y in zip(xs, ys)]

# build GeoDataFrame:
gdf = gpd.GeoDataFrame(
    {'id': range(len(points))},
    geometry=points,
    crs="EPSG:3413"
)

# export to disk:
gdf.to_file("./data/river_sections/section_coords.shp", 
            driver="ESRI Shapefile")
