# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:26:55 2020

@author: tsjlk
"""
# In[1]:

import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
del data['Unnamed: 32']

X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# In[2]:

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_2 = LabelEncoder()
y = labelencoder_X_2.fit_transform(y)

# In[3]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# In[4]:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# In[5]:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

def create_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 1024, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
    classifier.add(Dropout(0.1))
    return classifier

def add_layer(classifier, nums):
    for _ in range(nums):
        classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(0.1))
        
def get_result(classifier, layers):
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 1))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5).flatten()*1
    cm = confusion_matrix(y_test, y_pred)
    cm_explain = "Confusion_matrix of " + str(layers) + " layers model: \n", cm
    accuracy = 'Accuracy with ' + str(layers) + ' layers model is: ', (cm[0,0]+cm[1,1])/np.sum(cm)
    
    return (cm_explain, accuracy)

# In[6]:  
    
results = []
for layers in range(6):
    classifier = create_classifier()
    add_layer(classifier, layers)
    results.append(get_result(classifier, layers))
    
for cm, accuracy in results:
    print(cm)
    print(accuracy)
    print("\n")

