# -*- coding: utf-8 -*-
"""
Created on Sun May 10 02:50:10 2020

@author: tsjlk
"""

# In[1]:

import tensorflow as tf
tf.test.gpu_device_name()

# In[2]:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('S&P 500 Historical Data.csv')

dataset = data.iloc[:, 0:2].values
dataset[:,0] = pd.to_datetime(dataset[:,0])

plt.plot(dataset[:,0], dataset[:,1],  color = 'blue', label = 'Real S&P 500 Stock Price')
plt.show()

# In[3]:
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
#Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
dataset_scaled = sc.fit_transform(dataset[:,1].reshape(-1, 1))

# Visualising the scaled price
plt.plot(dataset[:,0], dataset_scaled,  color = 'blue', label = 'Real S&P 500 Stock Price')
plt.show()

# Creating a data structure, use 60 previous prices to price today's price
X = []
y = []
for i in range(90, len(dataset_scaled)):
    X.append(dataset_scaled[i-90:i-30, 0])
    y.append(dataset_scaled[i, 0])
X = np.array(X)
y = np.array(y)

# Reshaping
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
print(X.shape)

# In[4]:

# Split the data into training set and the test set in chronological order
from sklearn.model_selection import train_test_split
test_amount = int(X.shape[0]*0.7)
X_train = X[0:test_amount, :, :]
X_test = X[test_amount:, :, :]
y_train = y[0:test_amount]
y_test = y[test_amount:]

# In[5]:

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_absolute_error'])

# Show Model Structure
regressor.summary()

# Fitting the RNN to the Training set
history = regressor.fit(X_train, y_train, epochs = 50, batch_size = 32, 
                        validation_data=(X_test, y_test))

# In[6]:

# Visualize the training history
# Let's plot our results again:
mae = history.history['mean_absolute_error']
mae_test = history.history['val_mean_absolute_error']
epochs = range(len(mae))

from matplotlib import pyplot as plt
plt.plot(epochs, mae, 'b-', label='Training Data: MAE')
plt.plot(epochs, mae_test, 'r-', label='Test Data: MAE')
plt.title('Training and Test Dataset')
plt.legend()
plt.show()

# In[7]:

# Extract the real stock price in test set
real_stock_price = dataset[test_amount + 90:,1]
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(dataset[test_amount + 90:,0], real_stock_price, color = 'blue', label = 'Real S&P 500 Stock Price')
plt.plot(dataset[test_amount + 90:,0], predicted_stock_price, color = 'red', label = 'Predicted S&P 500 Stock Price')
plt.title('S&P 500 Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('S&P 500 Stock Price')
plt.xticks(rotation=70)
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse_test = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print('The RMSE error on the test dataset', rmse_test)