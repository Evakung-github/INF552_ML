# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:29:03 2020

@author: eva
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.loadtxt('classification.txt',delimiter=',')

import sklearn
sklearn.__version__
lg = LogisticRegression(solver = 'lbfgs',penalty='none')
lg.fit(X[:,:3],X[:,4])

print("accuracy rate:",lg.score(X[:,:3],X[:,4]))
print("coef:",lg.coef_)
print("intercept:",lg.intercept_)
print("# of iterations:",lg.n_iter_)
