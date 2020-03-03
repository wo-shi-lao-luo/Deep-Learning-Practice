# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:49:00 2020

@author: tsjlk
"""
# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

predictors = ["Age_08_04", "KM", "Fuel_Type", "HP", "Automatic", 
              "Doors", "Quarterly_Tax", "Mfr_Guarantee", "Guarantee_Period", 
              "Airco", "Automatic_airco", "CD_Player", "Powered_Windows", 
              "Sport_Model", "Tow_Bar"]

# Importing the dataset
dataset = pd.read_csv("ToyotaCorolla.csv")
features = dataset.loc[:, predictors].values
prices = dataset.loc[:, "Price"].values

print(features)
print(prices)

# In[2]
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("Fuel", OneHotEncoder(), [2])], remainder = 'passthrough')
features = ct.fit_transform(features)

# In[3]:

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, prices_train, prices_test = train_test_split(features, prices, test_size = 0.2, random_state = 0)

# In[4]:
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
sc = MinMaxScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)
prices_train = sc.fit_transform(DataFrame(prices_train))
prices_test = sc.transform(DataFrame(prices_test))

# In[5]:
# RMSE function
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(predictions, actuality):
    return np.sqrt(((predictions - actuality) ** 2).mean())

# In[6]:
# 1 hidden layer with 2 nodes

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(features[0])))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(features_train, prices_train, batch_size = 10, epochs = 20)

prices_pred = classifier.predict(features_test)
rmse1 = rmse(prices_pred, prices_test)
print(rmse1)

# In[7]:
# 1 hidden layer with 5 nodes

# Initialising the ANN
classifier2 = Sequential()

# Adding the input layer and the first hidden layer
classifier2.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(features[0])))

# Adding the output layer
classifier2.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier2.compile(optimizer='adam', loss="binary_crossentropy", metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier2.fit(features_train, prices_train, batch_size = 10, epochs = 20)

prices_pred2 = classifier2.predict(features_test)
rmse2 = rmse(prices_pred2, prices_test)
print(rmse2)

# In[8]:
# 2 hidden layer with 5 nodes

# Initialising the ANN
classifier3 = Sequential()

# Adding the input layer and the first hidden layer
classifier3.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(features[0])))

# Adding second hidden layer
classifier3.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier3.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier3.compile(optimizer='adam', loss="binary_crossentropy", metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier3.fit(features_train, prices_train, batch_size = 10, epochs = 20)

prices_pred3 = classifier3.predict(features_test)
rmse3 = rmse(prices_pred3, prices_test)
print(rmse3)

# In[9]:

#Result

print("1 hidden layer, 2 nodes, RMSE: ", rmse1)
print("1 hidden layer, 5 nodes, RMSE: ", rmse2)
print("2 hidden layer, 5 nodes, RMSE: ", rmse3)
print("Based on comparision of 3 different models, the model with 1 hidden layer and 5 nodes has the best performance")