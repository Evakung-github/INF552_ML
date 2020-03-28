# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:33:17 2020

@author: eva
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


class logistic:
    
    def __init__(self,X,y):
        self.X = X
        self.N = self.X.shape[0]
        self.X = np.hstack((np.ones((self.N,1)),self.X))
        self.y = y
        
    def sigmoid(self,z):
        return np.exp(z)/(1+np.exp(z))
    
    def Ein(self,z):
        sig = self.sigmoid(z)
        return -1/self.N*np.log(sig).sum()


    def fit(self,eta = 0.1,tol = 0.0001,max_iter=7000):
        self.iter = 0
        self.init_w = np.random.randint(-10,10,size = (self.X.shape[1]))
        self.loss = []
        self.accuracy = []
        prev_loss = float('inf')
        w = self.init_w
        diff = 1
        while self.iter<max_iter and abs(diff)>tol:
            z = self.X@w
            self.accuracy.append((np.sign(self.sigmoid(z)-0.5)==self.y).sum()/self.N)
            
            z = z * self.y
            
            l = self.Ein(z)
            self.loss.append(l)
            diff,prev_loss = l - prev_loss,l
            gradient_Ein = -1/self.N*(1/(1+np.exp(z)) * self.y)@self.X
            w = w -eta*gradient_Ein
            self.iter+=1
        self.best_estimate = w

       
    def iteration_result(self):
        fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)
        ax0.set_title('loss(Ein)')
        ax0.plot(range(len(self.loss)),np.array(self.loss),lw = 1)  
        ax1.set_title('accuracy')
        ax1.plot(range(len(self.accuracy)),np.array(self.accuracy),lw = 1)
        
    def predict(self,data):
        data = np.hstack((np.ones((2000,1)),data))
        z = data@self.best_estimate
        z = self.sigmoid(z)
        return np.sign(z-0.5)
        
    def plot_result(self,w = []):
        colors = list(mpl.colors.TABLEAU_COLORS)
        cmap = mpl.colors.ListedColormap(colors[:2])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        if list(w):
            xx, yy = np.meshgrid(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1))
            z = -(w[1]*xx+w[2]*yy+w[0])*1.0/w[3]
            ax.plot_surface(xx,yy,z,alpha = 0.6,color='tab:green')
    
        ax.scatter(self.X[:,1],self.X[:,2],self.X[:,3],c=self.y,cmap = cmap)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()        


X = np.loadtxt('classification.txt',delimiter=',')

model = logistic(X[:,:3],X[:,4])
model.fit(tol = 0.000000001)        
model.iteration_result()

model.plot_result(model.best_estimate)
result = model.predict(X[:,:3])
print("weights:",model.best_estimate)
print("accuracy rate: {:.3f}%".format(sum(result == model.y)/model.N*100))
print("# of iterations:",model.iter)
#a = model.X@model.best_estimate
#a = np.exp(a)/(1+np.exp(a))
#(np.sign(a - 0.5) == model.y).sum()/2000
#
#data = np.hstack((np.ones((2000,1)),X[:,:3]))
#model.predict(data) == np.sign(a - 0.5)






