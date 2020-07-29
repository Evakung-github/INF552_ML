# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:32:27 2020

@author: eva
"""

import numpy as np


class LinearRegressionUsingGD:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        self.cost_ = []
        self.w_ = np.array([ 2, -2,  2])
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = x@self.w_
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ = self.w_ - (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.w_)

#data = np.hstack((np.ones((3000,1)),X[:,:2]))    

a = LinearRegressionUsingGD()
a.fit(data,X[:,2])

a.w_


plt.plot(a.cost_)