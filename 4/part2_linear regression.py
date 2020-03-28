# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:58:04 2020

@author: eva
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

X = np.loadtxt('linear-regression.txt',delimiter=',')

LinearR = LinearRegression()
LinearR.fit(X[:,:2],X[:,2])

print("coef:",LinearR.coef_)
print("intercept:",LinearR.intercept_)


LinearR.predict(X[0,:2].reshape(1,-1))
X[0,2]

lasso = Lasso(alpha = 0.2)
lasso.fit(X[:,:2],X[:,2])


lasso.coef_
lasso.intercept_
lasso.n_iter_