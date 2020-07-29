# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:29:19 2020

@author: eva
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


X = np.loadtxt('classification.txt',delimiter=',')
N = X.shape[0]
X = np.hstack((np.ones((N,1)),X))




#learning rate
lr = 0.1
#initial weight
init_w = np.random.randint(0,25,size = (4))

def sign(w,x):
    return np.sign(w@x)


iter = 0
number_violated=[]
min_nv = N
optimal_w = init_w
w = init_w
violated_contraint = True
while violated_contraint and iter < 7000:
    violated_contraint = False
    a = np.sign(X[:,:4]@w)*X[:,5]
    nv = sum(a==-1)
    number_violated.append(nv)
    if nv < min_nv:
        min_nv = nv
        optimal_w = w
    if nv > 0:
        violated_contraint = True
        id = np.where(a == -1)[0][0]
        w = w + X[id,5] * lr * X[id,:4]
        iter += 1

        



np.where(a == 1)[0][0]

#check result
result = np.c_[X,np.sign(X[:,:4]@optimal_w)]
a = np.sign(X[:,:4]@optimal_w)*X[:,5]
len(np.where(a==1)[0])

#accuracy rate

accuracy = len(np.where(a==1)[0])/N

#plot each iteration
fig = plt.figure()
ax = fig.subplots()
ax.plot(range(len(number_violated)),np.array(number_violated),lw = 0.2)



#create plane
xx, yy = np.meshgrid(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1))
z = -(optimal_w[1]*xx+optimal_w[2]*yy)*1.0/optimal_w[3]

#Plot

colors = list(mpl.colors.TABLEAU_COLORS)
cmap = mpl.colors.ListedColormap(colors[:2])

#cdict = {1:'red',-1:'blue'}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#plot plane
ax.plot_surface(xx,yy,z,alpha = 0.6,color='tab:green')

#plot points
ax.scatter(X[:,1],X[:,2],X[:,3],c=X[:,5],cmap = cmap)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()