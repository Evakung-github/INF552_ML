# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:18:20 2020

@author: eva
"""

#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import numpy as np
from sklearn.decomposition import PCA

X = np.loadtxt('pca-data.txt')

def normalize(X):
    mu = X.mean(axis = 0)
    sigma = np.std(X,axis = 0)
    sigma_zero = sigma.copy()
    sigma[sigma_zero ==0 ] = 1.
    
    return (X-mu)/sigma


Xbar = normalize(X)


P = PCA(n_components=2, svd_solver='full')

r = P.fit_transform(Xbar)

sklearn_reconst = P.inverse_transform(r)

P.components_
P.explained_variance_



np.square(pmans - sklearn_reconst).sum()
