# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 23:46:28 2020

@author: eva
"""

import numpy as np
import random
import matplotlib.pyplot as plt

X = np.loadtxt('fastmap-data.txt')

Xb = pd.read_csv('fastmap-data.txt',delim_whitespace=True,header = None)
Xb[1][0]


pd.read_csv?

Distance_function = dict()

for i in X:
    Distance_function[(i[0]-1,i[1]-1)] = i[2]

nunique = len(set(X[:,0]))+1
new_dim = 2
cur_dim = 0
def old_d(i,j):
    if i == j:
        return 0
    return Distance_function[(min(i,j),max(i,j))]
    
def d(i,j,n):
    d = old_d(i,j)
    for c in range(n):
        d -= (P[i][c]-P[j][c])**2
    
    return d


def findMaxDpairs(cur_dim):
    init_b = random.randint(0,nunique-1)
    prev_b = -1
    #initial max distance
    m = -1
    while True:
        for i in range(nunique):
            if i == init_b:
                continue
            Di = d(i,init_b,cur_dim)
            if Di>m:
                m = Di
                a = i
        if a == init_b:
            break
        init_b,prev_b = a,init_b
        
    return a,prev_b,m



def p_dist(a,b,i,cur_dim):
    xi = (d(a,i,cur_dim)**2+d(a,b,cur_dim)**2-d(i,b,cur_dim)**2)/(2*d(a,b,cur_dim))
    return xi
    
    
def main():
    P = np.zeros((nunique,new_dim))
    for cur_dim in range(new_dim):
        a,b,m = findMaxDpairs(cur_dim)
        for i in range(nunique):
            xi = p_dist(a,b,i,cur_dim)
            P[i][cur_dim] = xi
    return P
    
P = main()



fig,ax = plt.subplots()

ax.scatter(P[:,0],P[:,1],label = P.index)


    
