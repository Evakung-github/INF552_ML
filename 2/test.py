# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:04:25 2020

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


#data.plot.scatter(x = "X",y="Y")

train = data.copy(deep=True)
train['cluster'] = 0



def initial_point(n,k):
    return random.sample(range(n),k)



def kmeans(train,k):
    initial = initial_point(len(train),k)
    
    prev_cluster = [[0,0] for i in range(k)]
    cur_cluster = list(map(lambda x:[train.iloc[x].X,train.iloc[x].Y],initial))

    count = 0
    while sorted(prev_cluster) != sorted(cur_cluster):
        print(count)
        for i in range(len(train)):
            distance = float('inf')
            m = k + 1
            for j in range(len(cur_cluster)):
                d = np.sqrt((train.iloc[i].X-cur_cluster[j][0]) ** 2 + (train.iloc[i].Y-cur_cluster[j][1]) ** 2)
                if d < distance:
                    distance = d
                    m = j
            train.iloc[i,2] = m
        
        prev_cluster,cur_cluster = cur_cluster,train.groupby('cluster').mean().values.tolist()
        count += 1
    return [train,cur_cluster]





def plot(result,centroid):
    k = len(centroid)
    colors = list(mpl.colors.TABLEAU_COLORS)
    cmap = mpl.colors.ListedColormap(colors[:k])
    fig,ax = plt.subplots()
    ax.scatter(result["X"],result["Y"],c = result['cluster'],cmap = cmap,s=20)
    i = 1
    for c in centroid:
        a = ax.scatter(c[0],c[1],marker = 'X',s = 100,c = 'black')#,label = 'centroid')# '+str(i))
        i+=1
    ax.legend([a],['centroid'])
    plt.show()


train = np.array(data).T

#result,centroid = kmeans(train,3)
#plot(result,centroid)
#
#    
#r = np.random.randint(low = 1000,size = (150,3))
#r = r.T / r.sum(axis = 1)
#
#rc = r.sum(axis = 1)
#
#(np.array(data)-mu).shape
#
#mu = np.array([(r[0]*data.X).mean(),(r[0]*data.Y).mean()])
#sigma = np.matmul(((r[0]*train).T-mu).T,((r[0]*train).T-mu))/150
#
#
#
#c = -0.5 * np.diag((train.T-mu)@np.linalg.inv(sigma)@(train.T-mu).T)
#p = 1/(2*np.pi)*(np.linalg.det(sigma)**-0.5)*np.exp(c)



def p_x_ci(train,mu,cov_matrix):
    dim = cov_matrix.shape[0]
    cov_inverse = np.linalg.inv(cov_matrix)
    
    c = -0.5 * np.diag((train.T-mu)@cov_inverse@(train.T-mu).T)
    p = ((2*np.pi)**(-dim/2))*(np.linalg.det(cov_matrix)**-0.5)*np.exp(c)
    return p

def p_c_xi(rc,p):
    p = rc * p.T
    deno = p.sum(axis = 1)
    return p.T/deno


def EM(train,k):
    initial = np.random.randint(low = 1000,size = (train.shape[1],k))
    pre_rc = np.array([0.3,0.2,0.5])
    r = initial.T / initial.sum(axis = 1)
    rc = r.sum(axis = 1)
    count = 0

    while sorted(pre_rc) != sorted(rc):
        print(count)
        
        p = []
        for i in range(k):
            mu = np.array([(r[i]*train[0]).mean(),(r[i]*train[1]).mean()])
            cov_matrix = np.matmul(((r[i]*train).T-mu).T,((r[i]*train).T-mu))/train.shape[1]
            p.append(p_x_ci(train,mu,cov_matrix))
        
        p = np.array(p)
        pre_rc,r = rc,p_c_xi(rc,p)
        rc = r.sum(axis = 1)
        count+=1
        if count >5:
            return r
    return r
    
r = EM(train,3)

r.argmax(axis = 0)

data['cluster'] = r.argmax(axis=0)
result = data

fig, ax = plt.subplots()
#result.plot.scatter(x = "X",y="Y",c = 'cluster',cmap = cmap,norm=norm)   
ax.scatter(result["X"],result["Y"],c = result['cluster'],cmap = cmap,s=20)
ax.scatter(centroid[0][0],centroid[0][1],marker = 'X',s = 100,label = 'centroid 1')
ax.scatter(centroid[1][0],centroid[1][1],marker = 'X',s = 100,label = 'centroid 2')
ax.scatter(centroid[2][0],centroid[2][1],marker = 'X',s = 100,label = 'centroid 3')
ax.legend(('X')('centroid'))

#fig.colorbar(ax.collections[0], ticks=np.linspace(0,2,3))

