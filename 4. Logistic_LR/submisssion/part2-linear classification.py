# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:37:31 2020

@author: eva
"""
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt

X = np.loadtxt('classification.txt',delimiter=',')


#perceptron
per = Perceptron()
per.fit(X[:,:3],X[:,4])

np.hstack([per.intercept_,per.coef_[0]])
per.coef_

per.n_iter_
per.score(X[:,:3],X[:,4])


#sgdclassifier

sgd = SGDClassifier(tol = None,loss="perceptron", eta0=0.1, learning_rate="constant", penalty=None)
#sgd = SGDClassifier(penalty=None)
sgd.fit(X[:,:3],X[:,4])
print("weights of sgd:", np.hstack([sgd.intercept_,sgd.coef_[0]]))
print("accuracy:",sgd.score(X[:,:3],X[:,4]))

sgd.n_iter_


def plot_result(X,y,w = []):
    colors = list(mpl.colors.TABLEAU_COLORS)
    cmap = mpl.colors.ListedColormap(colors[:2])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if list(w):
        xx, yy = np.meshgrid(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1))
        z = -(w[1]*xx+w[2]*yy+w[0])*1.0/w[3]
        ax.plot_surface(xx,yy,z,alpha = 0.6,color='tab:green')

    ax.scatter(X[:,0],X[:,1],X[:,2],c=y,cmap = cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

plot_result(X[:,:3],X[:,4],np.hstack([sgd.intercept_,sgd.coef_[0]]))