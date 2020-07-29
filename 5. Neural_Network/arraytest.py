# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:51:33 2020

@author: eva
"""

a = np.array([[1,1,2],[5,3,4]])
b = np.array([[2,3],[1,1]])



np.multiply(a.T,b)

np.multiply(a[0].reshape(-1,1),b[0])
t = []
t.append(a[0].reshape(-1,1)*b[0])


np.array(t).mean(axis = 0).shape



np.random.randint(low = 10,size=(10))