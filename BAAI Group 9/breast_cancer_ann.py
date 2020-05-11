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
from sklearn.preprocessing import StandardScaler

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

# In[4]:

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

def create_classifier(nodes):
    classifier = Sequential()
    classifier.add(Dense(units = nodes, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
    classifier.add(Dropout(0.1))
    return classifier

def add_layer(classifier, layers, nodes):
    for _ in range(layers):
        classifier.add(Dense(units = nodes//2, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(0.1))
        
def get_result(classifier, layers, nodes, X_train, X_test, y_train, y_test):
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 1))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5).flatten()*1
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/np.sum(cm)
    return accuracy

# In[5]:  
# Testing different combinations

nodes_set = [8, 16, 32, 64, 128, 256, 512, 1024]
hidden_layer_limit = 7
results = []
test_amount = 100

for nodes in nodes_set:
    for layers in range(hidden_layer_limit):
        temp_accuracy_sum = 0
        for count in range(test_amount):
            print('start testing ANN with %d hidden layers, %d/%d nodes' %(layers, nodes, nodes//2))
            print('Test %d' %(count+1))
            X_train, X_test, y_train, y_test = split_data(X, y)
            classifier = create_classifier(nodes)
            add_layer(classifier, layers, nodes)
            temp_accuracy_sum += get_result(classifier, layers, nodes, X_train, X_test, y_train, y_test)
        average_accuracy = temp_accuracy_sum / test_amount
        print('Average accuracy of ANN with %d hidden layers, %d/%d nodes: %f' %(layers, nodes, nodes//2, average_accuracy))
        results.append((average_accuracy, layers, nodes))
        
print(results)

# In[6]:
# Get the top 5 accurate combination

results.sort(key=lambda x: -x[0])

top_5 = results[:5]

print(top_5)

for i, result in enumerate(top_5):
    accuracy, layers, nodes = result
    print("The Top %d ANN model has %d hidden layers." %(i + 1, layers))
    print("The input layer has %d nodes, and all the hidden layers have %d nodes" %(nodes, nodes//2))
    print("The accuracy of this model is: %f" %accuracy)
    
# In[7]:
worst_5 = results[-5:]

print(worst_5)

for i, result in enumerate(worst_5):
    accuracy, layers, nodes = result
    print("The Worst %d ANN model has %d hidden layers." %(i + 1, layers))
    print("The input layer has %d nodes, and all the hidden layers have %d nodes" %(nodes, nodes//2))
    print("The accuracy of this model is: %f" %accuracy)