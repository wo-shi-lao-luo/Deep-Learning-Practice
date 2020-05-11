# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:47:44 2020

@author: imba_ieva
"""

#%%
"""
0. Load packages
"""
import numpy as np
import pandas as pd
import seaborn as sns

#%%
"""
1. Load CSV file
"""
data = pd.read_csv("C:/Users/imba_ieva/Desktop/AI Final Project/data.csv")

data.info()
data.head()

#%%
"""
2. Data cleaning
"""

#2.1 Dropping irrelevent column 
data.drop(['id','Unnamed: 32'], axis=1, inplace=True)
data.head()

#2.2 Converting the diagnosis value of M and B to binary 1 and 0
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data.head()

#%%
"""
3. Data Visulization
"""
# 1. Frequency of cancer stages 
sns.countplot(data['diagnosis'],label="Count")# from this graph we can see that there is a more number of bengin stage of cancer which can be cure

#%%
"""
4. Splitting Data for KNN and RandomForest
"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

X = np.array(data.iloc[:,1:])
y = np.array(data['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 12345)

#%%
"""
5. KNN
"""
# 1.Fitting training dataset for KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit (X_train,y_train)

# 2.Evaluating KNN model on test set
prediction_knn=knn.predict(X_test)
accuracy_knn = knn.score(X_test,y_test)
print("Accuracy : %s" % "{0:.3%}".format(accuracy_knn))
from sklearn import metrics
#evaluation(Confusion Metrix)
print("Confusion Metrix for KNN:\n",metrics.confusion_matrix(prediction_knn,y_test))

# 3. Optimizing for number of K by calculating RMSE
from math import sqrt
from sklearn.metrics import mean_squared_error

rmse_val = []
for K in range(20):
    K = K+1
    knn = KNeighborsClassifier(n_neighbors = K)

    knn.fit(X_train, y_train)  #fit the model
    prediction_knn=knn.predict(X_test)
    error = sqrt(mean_squared_error(y_test,prediction_knn)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
    
# Plot for best K value
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot().set_title('RMSE')


#%%
"""
6. Random Forest 
"""
rf = RandomForestClassifier(n_estimators=100)

# 1. Fitting training set into Random Forest model
rf.fit(X_train,y_train)

# 2. Evaluating RF model
prediction_rf=rf.predict(X_test)
accuracy_rf = rf.score(X_test,y_test)
print("Accuracy : %s" % "{0:.3%}".format(accuracy_rf))

from sklearn import metrics
#evaluation(Confusion Metrix)
print("Confusion Metrix for Random Forest:\n",metrics.confusion_matrix(prediction_rf,y_test))
