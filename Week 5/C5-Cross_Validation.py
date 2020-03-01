# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:34:42 2019

@author: wchen
"""
# In[1]:
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  

from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()

boston['MEDV'] = boston_dataset.target
  
# In[2]:
# Pick Two Predictors and Target
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']


# In[3]:
# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# In[4]:
# Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# In[5]:
# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

# In[6]:
# K-fold cross validation
K = 100
rmse_train_array = np.zeros(K)
rmse_test_array = np.zeros(K)
for fold in range(K):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=fold)
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)
    y_train_predict = lin_model.predict(X_train)
    rmse_train = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    y_test_predict = lin_model.predict(X_test)
    rmse_test = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    rmse_train_array[fold] = rmse_train
    rmse_test_array[fold] = rmse_test
    
import seaborn as sns 
sns.distplot(rmse_test_array, bins=20, kde=False)
print(np.mean(rmse_test_array))
