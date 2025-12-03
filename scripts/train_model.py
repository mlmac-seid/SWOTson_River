# -*- coding: utf-8 -*-
"""
created on: 2025-11-18
@author:    Jasper Heuer
use:        train models to predict Watson River land cover
"""

# import packages =============================================================

import os
import pickle
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# set working directory =======================================================

base_path = "C:/Jasper/PhD/Classes/ECS_851_Geospatial_Data_Science/Project"
os.chdir(base_path)

# read data ===================================================================

df = pd.read_csv("./data/training_data/train_dataset.csv")
df = df.drop(columns=["Unnamed: 0", "x", "y", "landcover"])

X = df.drop(columns="int_lc")
y = df["int_lc"]

# train-test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# build RandomForest model ====================================================

# cross-validation score:
kf = KFold(n_splits=5, shuffle=True, random_state=42)

clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=kf)

print(f"Random Forest score: {np.mean(scores)*100:.2f}%")

# OOB-score:
clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
clf.fit(X, y)

print(f"OOB score: {clf.oob_score_*100:.2f}%")

# train model on training data only:
clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
clf.fit(X_train, y_train)

# score model:
results = clf.score(X_test, y_test)
print(f"Model score (testing data): {results*100:.2f}%")

# save model:
with open("./data/model/RF_IMG_DEM_NDWI_NDVI.pkl","wb") as f:
    pickle.dump(clf,f)
