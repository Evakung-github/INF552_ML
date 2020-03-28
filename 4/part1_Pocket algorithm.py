# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:10:54 2020

@author: eva
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


class perceptron_pocket:
    
    def __init__(self,X,y):
        self.X = X
        self.N = self.X.shape[0]
        self.X = np.hstack((np.ones((self.N,1)),self.X))
        self.y = y
                   
    
    def fit(self,lr,max_iter):
        self.lr = lr
        self.init_w = np.random.randint(-10,10,size = (self.X.shape[1]))
        self.iter = 0
        self.number_violated = []
        self.min_nv = self.N
        self.optimal_w = self.init_w
        w = self.init_w
        violated_contraint = True
        while violated_contraint and self.iter < max_iter:
            violated_contraint = False
            a = np.sign(self.X@w)*self.y
            nv = sum(a==-1)
            self.number_violated.append(nv)
            if nv < self.min_nv:
                self.min_nv = nv
                self.optimal_w = w
            if nv > 0:
                violated_contraint = True
                id = np.where(a == -1)[0][0]
                w = w + self.y[id] * self.lr * self.X[id]
                self.iter += 1
    
    def accuracy(self):
        return 1 - self.min_nv/self.N 
    
    def iteration_result(self):
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(range(len(self.number_violated)),np.array(self.number_violated),lw = 0.2)
        ax.set_ylabel("Number of misclassifications")
        
    
    def plot_result(self,w = []):
        colors = list(mpl.colors.TABLEAU_COLORS)
        cmap = mpl.colors.ListedColormap(colors[:2])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        if list(w):
#            if round(w[0],8) != 0:
#                print("coefficient of constant is not small enough to ignore, please check")
#            else:
            xx, yy = np.meshgrid(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1))
            z = -(w[1]*xx+w[2]*yy+w[0])*1.0/w[3]
            ax.plot_surface(xx,yy,z,alpha = 0.6,color='tab:green')
    
        ax.scatter(self.X[:,1],self.X[:,2],self.X[:,3],c=self.y,cmap = cmap)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        
        
        
X = np.loadtxt('classification.txt',delimiter=',')

model = perceptron_pocket(X[:,:3],X[:,4])
model.fit(0.1,7000)
print("accuracy:",model.accuracy())
print("optimal_w:",model.optimal_w)
print("# of iterations:",model.iter)
model.min_nv/model.N
model.iteration_result()
model.plot_result(model.optimal_w)

a = model.optimal_w
