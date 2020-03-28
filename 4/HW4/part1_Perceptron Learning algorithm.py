# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:24:56 2020

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
init_w = np.random.randint(0,50,size = (4))

def sign(w,x):
    return np.sign(w@x)

w = init_w
violated_contraint = True
while violated_contraint:
    violated_contraint = False
    for i in range(N):
        if sign(w,X[i,:4])*X[i,4] == -1:
            w = w + X[i,4] * lr * X[i,:4]
            violated_contraint = True
            break
        

#check result
result = np.c_[X,np.sign(X[:,:4]@w)]
a = np.sign(X[:,:4]@w)*X[:,4]
len(np.where(a==1)[0])

#accuracy rate

accuracy = len(np.where(a==1)[0])/N
print("accuracy:",accuracy)
print("optimal_w:",w)


#create plane
xx, yy = np.meshgrid(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1))
z = -(w[1]*xx+w[2]*yy+w[0])*1.0/w[3]

#Plot

colors = list(mpl.colors.TABLEAU_COLORS)
cmap = mpl.colors.ListedColormap(colors[:2])

#cdict = {1:'red',-1:'blue'}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#plot plane
ax.plot_surface(xx,yy,z,alpha = 0.6,color='tab:green')

#plot points
ax.scatter(X[:,1],X[:,2],X[:,3],c=X[:,4],cmap = cmap)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
