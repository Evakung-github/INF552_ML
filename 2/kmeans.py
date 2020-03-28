# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:36:40 2020

@author: eva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

data = pd.read_csv("clusters.txt",header = None)
data.columns = ["X","Y"]
#cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen"])
#norm = mpl.colors.BoundaryNorm(np.arange(-0.5,3), cmap.N) 


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
        #print(count)
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

k = kmeans(train,3)
k[1]


def plot(result,centroid):
    k = len(centroid)
    colors = list(mpl.colors.TABLEAU_COLORS)
    cmap = mpl.colors.ListedColormap(colors[:k])
    fig,ax = plt.subplots()
    ax.scatter(result["X"],result["Y"],c = result['cluster'],cmap = cmap,s=20)
    i = 1
    for c in centroid:
        a = ax.scatter(c[0],c[1],marker = 'X',s = 100,c = 'black')#,label = 'centroid')# '+str(i))
        label = "["+str(round(c[0],2))+","+str(round(c[1],2))+"]"
        ax.annotate(label, # this is the text
                    (c[0],c[1]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
        i+=1
    ax.legend([a],['centroid'])
    plt.show()
    
result,center = kmeans(train,3)

plot(result,center)