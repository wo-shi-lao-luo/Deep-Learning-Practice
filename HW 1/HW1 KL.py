# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:49:00 2020

@author: tsjlk
"""
# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ToyotaCorolla.csv').iloc[:,1:]
features = dataset.iloc[:, dataset.columns != "Price"].values
value = dataset.iloc[:, 1].values

print(features[:])
print(value)

# In[2]:

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Convert 'Diesel' to 1, 'Petrol' to 0
labelencoder_X_2 = LabelEncoder()
fuel = features[:, 4]
print(fuel)
fuel = labelencoder_X_2.fit_transform(X[:, 4])
print(fuel)

# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)