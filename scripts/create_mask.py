# -*- coding: utf-8 -*-
"""
created on: 2025-10-30
@author:    Jasper Heuer
use:        create mask for Watson river basin
"""

"""
Add creation of lat-lon masks"""

# import packages =============================================================

import os

# import own utils module =====================================================

base_path = "C:/Jasper/PhD/Projects/Supraglacial_lakes/src/"
os.chdir(base_path)

import utils

# set working directory =======================================================

base_path = "C:/Jasper/PhD/Classes/ECS_851_Geospatial_Data_Science/Project/"
os.chdir(base_path)

# define study area ===========================================================

res = 10
crs = "EPSG:3413"
bounds = (-248791, -2510840, -225623, -2503293)

# create mask =================================================================

utils.create_mask("./data/masks/", bounds, res, crs)

