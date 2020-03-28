# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:46:45 2020

@author: eva
"""
#https://stackoverflow.com/questions/44765682/in-sklearn-decomposition-pca-why-are-components-negative

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X = np.loadtxt('pca-data.txt')

class PCA:
    def __init__(self,X,components):
        self.X = X
        self.components = components
        mu = X.mean(axis = 0)
        sigma = np.std(X,axis = 0)
        sigma_zero = sigma.copy()
        sigma[sigma_zero ==0 ] = 1.
        self.Xbar = (X-mu)/sigma
        self.cov = (self.Xbar.T@self.Xbar)/len(self.Xbar)
        
        
    def getEigen(self):
        eig_value,eig_vector = np.linalg.eig(self.cov)
        sort = eig_value.argsort()[::-1]
        self.eig_value = eig_value[sort]
        self.eig_vector = eig_vector[:,sort]
        return (self.eig_value,self.eig_vector)
    
    



    def fit(self):
        self.getEigen()
        pc = self.eig_vector[:,:self.components]
        
        return (pc.T@self.Xbar.T).T
    
    
    def reconstruct(self):
        #projection matrix
        pc = self.eig_vector[:,:self.components]
        pm = pc @ np.linalg.inv(pc.T@pc) @ pc.T
        reconstruct_X = (pm @ self.Xbar.T).T
        return reconstruct_X




model = PCA(X,2)
result = model.fit()
pmans = model.reconstruct()



model.eig_vector[:,:2]
model.cov









fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result[:,0],result[:,1],marker = 'o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

