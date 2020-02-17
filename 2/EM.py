# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:38:23 2020

@author: eva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

data = pd.read_csv("clusters.txt",header = None)
data.columns = ["X","Y"]
cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen"])
norm = mpl.colors.BoundaryNorm(np.arange(-0.5,3), cmap.N) 

initial = np.random.randint(low = 1000,size = (train.shape[0],3))
pre_rc = np.array([0.3,0.2,0.5])
r = initial.T / initial.sum(axis = 1)

def p_x_ci(train,mu,cov_matrix):
    """
    train: df-like, shape = (n_samples,n_features)
    mu   : array-like, shape = (n_features,)
    cov  : array-like, shape = (n_features,n_features)
    """
    
    dim = cov_matrix.shape[0]
    cov_inverse = np.linalg.inv(cov_matrix)
    train = np.array(train)
    c = -0.5 * np.diag((train-mu)@cov_inverse@(train-mu).T)
    p = ((2*np.pi)**(-dim/2))*(np.linalg.det(cov_matrix)**-0.5)*np.exp(c)
    return p

p_x_ci(np.array([[1,2]]),a,b)



def p_c_xi(rc,p):
    """
    rc   : array-like, shape = (n_features,)
    p    : array-like, shape = (n_features,n_samples)
    """
    p = rc * p.T
    deno = p.sum(axis = 1)
    return p.T/deno

def compute_mu_cov(train,r):
    """
    train: df-like, shape = (n_samples,n_features)
    r    : array-like, shape = (n_samples,)
    """
    
    mu = []
    D = []
    r_diag = np.diag(r)

    for feature in range(train.shape[1]):
        m = (r*train.iloc[:,feature]).sum()/r.sum()
        mu.append(m)
        D.append((train.iloc[:,feature]-m))
    D = np.array(D)
    cov_matrix = (D@r_diag@D.T)/r.sum()
    #cov_matrix = (r*train.T)@(r*train.T).T/train.shape[0]

    return [mu,cov_matrix]



def compute_ln_p(p,rc):
    """
    rc   : array-like, shape = (n_features,)
    p    : array-like, shape = (n_features,n_samples)
    """    
    p = rc * p.T
    log = np.log(p.sum(axis=0))
    return log.sum()
    


np.cov(data.T,ddof = 0)
test = pd.DataFrame([[1,2,3],[3,4,6]])    
a,b =compute_mu_cov(data,np.array([1]*150))


def EM(train,k):
    """
    train: df-like, shape = (n_samples,n_features)
    """
    initial = np.random.randint(low = 1000,size = (train.shape[0],k))
    #pre_rc = np.array([0.3,0.2,0.5])
    r = initial.T / initial.sum(axis = 1)
    rc = r.sum(axis = 1)
    count = 0
    pre_mlh = float('inf')
    mlh = 0
    while abs(pre_mlh-mlh)>0.001:
        print(count)
        m = []
        c = []

        p = []
        for i in range(k):
            mu,cov_matrix = compute_mu_cov(train,r[i])
            p.append(p_x_ci(train,mu,cov_matrix))
            m.append(mu)
            c.append(cov_matrix)
        
        p = np.array(p)
        pre_mlh,mlh = mlh,compute_ln_p(p,rc)
        r = p_c_xi(rc,p)
        rc = r.sum(axis = 1)/150
        count+=1
        print(mlh)
#        if count >1:
#            return m,c,rc,r
    return r

result = data
h = EM(data[['X','Y']],3)




result['cluster'] = h.argmax(axis = 0)


fig, ax = plt.subplots()
#result.plot.scatter(x = "X",y="Y",c = 'cluster',cmap = cmap,norm=norm)   
ax.scatter(result["X"],result["Y"],c = result['cluster'],cmap = cmap,s=20)



