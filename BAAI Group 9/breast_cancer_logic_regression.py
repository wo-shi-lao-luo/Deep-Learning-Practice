# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:29:15 2020

@author: yz514312
"""
# In[1]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('data.csv')
del df['Unnamed: 32']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler 
 
x = df.drop('diagnosis', axis=1)
y = df['diagnosis']


# In[2]:

def split_data(x, y):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    sc_x = StandardScaler() 
    x_train = sc_x.fit_transform(x_train) 
    x_test = sc_x.transform(x_test) 
    
    return x_train, x_test, y_train, y_test

def logistic_regression(x_train, x_test, y_train, y_test):
    
    classifier = LogisticRegression(random_state = 0) 
    classifier.fit(x_train, y_train) 
    
    y_pred = classifier.predict(x_test) 
     
    from sklearn.metrics import confusion_matrix 
    cm = confusion_matrix(y_test, y_pred) 
    
    print ("Confusion Matrix : \n", cm) 
    
    from sklearn.metrics import accuracy_score 
#    accuracy = accuracy_score(y_test, y_pred)
    accuracy = (cm[0,0]+cm[1,1])/np.sum(cm)
    print ("Accuracy : ", accuracy) 
    
    return accuracy


# In[3]:

x_train, x_test, y_train, y_test = split_data(x, y)
logistic_regression(x_train, x_test, y_train, y_test)
