# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:57:28 2020

@author: eva
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.loadtxt('fastmap-data.txt')

labels = pd.read_csv("fastmap-wordlist.txt",header = None).values

class fastmap:
    def __init__(self,X,new_dim):
        self.Distance_function = dict()
        for i in X:
            self.Distance_function[(i[0]-1,i[1]-1)] = i[2]
        #number of object
        self.nunique = len(set(X[:,0]))+1
        #number of new dimensions
        self.new_dim = new_dim
        #track current dimension for calculating distances and store in P
        self.cur_dim = 0
        #array to store result
        self.P = np.zeros((self.nunique,new_dim))
        #list to store max pair used in each iteration
        self.maxPair = []
        
    def d(self,i,j):
        """
        input:  Two objects
        return: Distance that hasn't been counted in previous axis.
        """      
        if i == j:
            return 0
        d = (self.Distance_function[(min(i,j),max(i,j))])**2
        for c in range(self.cur_dim):
            d -= (self.P[i][c]-self.P[j][c])**2
        return d**0.5

    def findMaxDpairs(self):
        init_b = random.randint(0,self.nunique-1)
        prev_b = -1
        #initial max distance
        m = -1
        while True:
            for i in range(self.nunique):
                Di = self.d(i,init_b)
                if Di>m:
                    m = Di
                    a = i
            #If m doesn't change, it means that the previous pair has the max distance.
            if a == init_b:
                break
            init_b,prev_b = a,init_b
            
        return a,prev_b,m    
    
    def p_dist(self,a,b,i):
        """
        input:  max pair(a,b) and object i
        return: Distance between i and a when projecting i onto ab line
        """
        xi = (self.d(a,i)**2+self.d(a,b)**2-self.d(i,b)**2)/(2*self.d(a,b))
        return xi

    def main(self):
        """
        For each iteration, find the pair with maximum distance, then project every objects on to the maximum line.
        Calculate the distance between the projection and a, and the value is the coordinate of that axis.
        """
        while self.cur_dim<self.new_dim:
            a,b,m = self.findMaxDpairs()
            self.maxPair.append((a,b))
            for i in range(self.nunique):
                xi = self.p_dist(a,b,i)
                self.P[i][self.cur_dim] = xi
            self.cur_dim+=1
        return self.P     


#Implementing fastmap
P = fastmap(X,2)
P.findMaxDpairs()

P.main()
result = P.P

#Draw results

fig,ax = plt.subplots()
ax.scatter(result[:,0],result[:,1])

for i,txt in enumerate(labels):
    ax.annotate(txt[0],(result[i][0],result[i][1]),size = 14)

P.maxPair
#Calculate new distances of each pair in result.
new_d = np.zeros((45,3))

k = 0
for i in range(P.nunique):
    for j in range(i+1,P.nunique):
        d = np.sqrt(((result[i]-result[j])**2).sum())
        new_d[k][0],new_d[k][1],new_d[k][2] = i+1,j+1,d
        k+=1
        
a = new_d[:,2]-X[:,2]
np.where(a==0,a,1).sum()
np.where(a==0)

#Draw original position

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2])
ax.set_xlabel('Object1')
ax.set_ylabel('Object2')
ax.set_zlabel('Distance')
plt.show()





