# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:38:29 2020

@author: eva
"""

from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense



model = Sequential()
model.add(Dense(100, input_dim=960, 
	activation="sigmoid"))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])


model.fit(trainX,trainY,epochs=100,batch_size=10,verbose = False)

model.evaluate(trainX,trainY)
