# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:02:02 2020
@author: wchen
Business Application of A.I. Lecture 2: Python and Pandas
"""

# In[01]:
# Arithmetic operations:
a = 5
b = 10
c = a + b
d = a - b
e = a * b
f = a / b
g = b ** a
print(a, b, c, d, e, f, g)

# In[02]:
import numpy as np

x = np.array([1, 2, 3, 4])
sum1 = 0
for i in range(len(x)):
    sum1 += x[i]
Mean = sum1 / len(x)
print('Mean = ', Mean)

# In[03]:
# use numpy.mean() to calculate the mean
Mean2 = np.mean(x)

# if checking
if Mean == Mean2:
    print('My code is correct!')

# In[04]:
# while loop
import numpy as np

x = np.array([1, 2, 3, 4])
sum1 = 0
i = 0
while i < len(x):
    sum1 += x[i]
    i += 1
Mean = sum1 / len(x)
print('Mean=', Mean)

# In[05]:
# create your own function
import numpy as np


def my_mean_fun(data):
    sum1 = 0
    for i in range(len(data)):
        sum1 += data[i]
    Mean = sum1 / len(data)
    return (Mean)


x = np.array([1, 2, 3, 4])
Mean = my_mean_fun(x)
print('Mean = ', Mean)

# In[06]:
# Import pandas Library
import pandas as pd
import os
# Import data files
sfo_cust_data_2014 = pd.read_excel('sfo cust sat 2014 data file_WEIGHTED_flysfo.xlsx', 'Sheet 1')
print(sfo_cust_data_2014)

# import os

# os.chdir('D:/#DevCourses-GWU/#4_Deep_Learning')

# In[07]:
print(sfo_cust_data_2014.columns)

# In[08]:
sfo_cust_data_2014_subset = sfo_cust_data_2014[
    ['RESPNUM',
     'Q16LIVE',
     'Q7ART',
     'Q7FOOD',
     'Q7STORE',
     'Q7SIGN',
     'Q7WALKWAYS',
     'Q7SCREENS',
     'Q7INFODOWN',
     'Q7INFOUP',
     'Q7WIFI',
     'Q7ROADS',
     'Q7PARK',
     'Q7AIRTRAIN',
     'Q7LTPARKING',
     'Q7RENTAL',
     'Q7ALL',
     'Q9BOARDING',
     'Q9AIRTRAIN',
     'Q9RENTAL',
     'Q9FOOD',
     'Q9RESTROOM',
     'Q9ALL',
     'Q10SAFE',
     'Q12PRECHECKRATE',
     'Q13GETRATE',
     'Q14FIND',
     'Q14PASSTHRU'
     ]]
print(sfo_cust_data_2014_subset.columns)

# In[09]:
print(sfo_cust_data_2014_subset.dtypes)

# In[10]:
print(sfo_cust_data_2014_subset.shape)

# In[11]:
print(sfo_cust_data_2014.Q19GENDER.value_counts())

# In[12]:
# Make the gender column meaningful
sfo_cust_data_2014[['Q19GENDER']] = sfo_cust_data_2014[['Q19GENDER']].replace([1], 'Male')
sfo_cust_data_2014[['Q19GENDER']] = sfo_cust_data_2014[['Q19GENDER']].replace([2], 'Female')
sfo_cust_data_2014[['Q19GENDER']] = sfo_cust_data_2014[['Q19GENDER']].replace([3], 'Other')
sfo_cust_data_2014[['Q19GENDER']] = sfo_cust_data_2014[['Q19GENDER']].replace([0], 'Blank')

print(sfo_cust_data_2014.Q19GENDER.value_counts())

# In[13]:
sfo_cust_data_2014.to_excel("New_sfo_cust_data_2014.xlsx", sheet_name='Sheet2', index=False)

# In[14]:
sfo_cust_data_2015 = pd.read_csv('sfo cust sat 2015_data file_final_WEIGHTED_flysfo.csv')
print(sfo_cust_data_2015)

# In[15]:
sfo_cust_data_2015[['Location 1']] = sfo_cust_data_2015[['Location 1']].fillna(0)
print(sfo_cust_data_2015.head())

# In[16]:
sfo_cust_data_2015 = sfo_cust_data_2015.assign(YEAR=2015)
print(sfo_cust_data_2015)

# In[17]:
result = sfo_cust_data_2014[['Q16LIVE',
                             'Q7ART',
                             'Q7FOOD',
                             'Q7STORE',
                             'Q7SIGN',
                             'Q7WALKWAYS',
                             'Q7SCREENS',
                             'Q7INFODOWN',
                             'Q7INFOUP',
                             'Q7WIFI',
                             'Q7ROADS',
                             'Q7PARK',
                             'Q7AIRTRAIN',
                             'Q7LTPARKING',
                             'Q7RENTAL',
                             'Q7ALL',
                             'Q9BOARDING',
                             'Q9AIRTRAIN',
                             'Q9RENTAL',
                             'Q9FOOD',
                             'Q9RESTROOM',
                             'Q9ALL',
                             'Q10SAFE',
                             'Q12PRECHECKRATE',
                             'Q13GETRATE',
                             'Q14FIND',
                             'Q14PASSTHRU']].apply(pd.value_counts)
print(result)

# In[18]:
sfo_cust_data_2015.to_csv('New_sfo_cust_data_2015.csv', index=False)