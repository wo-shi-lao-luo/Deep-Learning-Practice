# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 01:40:40 2020

@author: tsjlk
"""
# In[1]

import tensorflow as tf
tf.test.gpu_device_name()

# In[2]:

import pandas as pd

data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', header=None) 

X = data.iloc[1:, 4].values
y = data.iloc[1:, 5].values

# In[3]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# In[4]:

import numpy as np
glove_file = 'glove.6B.50d.txt'   

with open(glove_file, 'r', encoding="utf8") as f:
    word_to_vec_map = {}
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

# In[5]:

import gensim
dataTrain_as_lists_of_words = []
for i in range(len(X_train)):
    a_piece_of_sentence = str(X_train[i])
    single_sentence_as_list_of_words = gensim.utils.simple_preprocess( a_piece_of_sentence ) 
    dataTrain_as_lists_of_words.append(single_sentence_as_list_of_words)
    
dataTest_as_lists_of_words = []
for i in range(len(X_test)):
    a_piece_of_sentence = str(X_test[i])
    single_sentence_as_list_of_words = gensim.utils.simple_preprocess( a_piece_of_sentence ) 
    dataTest_as_lists_of_words.append(single_sentence_as_list_of_words) 
    
# In[6]:

SENTENCE_LENGTH = 100
EMBEDDED_VECTOR_DIM = 50

list_of_words = dataTrain_as_lists_of_words[0]

sentense_word2vec = np.zeros((SENTENCE_LENGTH, EMBEDDED_VECTOR_DIM))

for word_nr in range( min(SENTENCE_LENGTH, len(list_of_words)) ):    
    word = list_of_words[word_nr]
    try:
        word_vec = word_to_vec_map[word]  
        sentense_word2vec[word_nr,:] = word_vec
    except:
        sentense_word2vec[word_nr,:] = np.zeros((EMBEDDED_VECTOR_DIM))
        
print(sentense_word2vec)

# In[7]:

trainX = []       
 
for sentence_nr in range(len(dataTrain_as_lists_of_words)):
    
    list_of_words = dataTrain_as_lists_of_words[sentence_nr]
    sentense_word2vec = np.zeros((SENTENCE_LENGTH, EMBEDDED_VECTOR_DIM))   
    
    for word_nr in range( min(SENTENCE_LENGTH, len(list_of_words)) ): 
        word = list_of_words[word_nr]
        try:
            word_vec = word_to_vec_map[word]
            sentense_word2vec[word_nr,:] = word_vec
        except:
            sentense_word2vec[word_nr,:] = np.zeros((EMBEDDED_VECTOR_DIM))
       
    trainX.append(sentense_word2vec)
 
trainX = np.array(trainX)
print(trainX.shape) 

trainY = np.array(y_train)
trainY = trainY.astype(int)
print(trainY.shape)
# In[8]:

testX = []        
for sentence_nr in range(len(dataTest_as_lists_of_words)):
    
    list_of_words = dataTest_as_lists_of_words[sentence_nr]
    sentense_word2vec = np.zeros((SENTENCE_LENGTH, EMBEDDED_VECTOR_DIM))
    
    for word_nr in range( min(SENTENCE_LENGTH, len(list_of_words)) ):  
        word = list_of_words[word_nr] 
        try:
            word_vec = word_to_vec_map[word]
            sentense_word2vec[word_nr,:] = word_vec
        except:
            sentense_word2vec[word_nr,:] = np.zeros((EMBEDDED_VECTOR_DIM))
         
    testX.append(sentense_word2vec)
 
testX = np.array(testX)
print(testX.shape) 

testY = np.array(y_test) 
testY = testY.astype(int)
print(testY.shape) 

# In[9]:

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()

model.add(LSTM(100, input_shape=(SENTENCE_LENGTH, EMBEDDED_VECTOR_DIM)))
model.add(Dense(8, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])    
model.summary()

history = model.fit(trainX,
          trainY,
          epochs=20,
          batch_size=32,
          verbose=1,
          validation_data=(testX, testY))

predY = model.predict(testX)
predY = np.argmax(predY, axis=1) 

# In[10]:

import matplotlib.pyplot as plt
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# In[11]:

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

print("The Root Mean Square Error is:", np.sqrt(mse(predY, testY)))
print("The Mean Absolute Error is:", mae(predY, testY))