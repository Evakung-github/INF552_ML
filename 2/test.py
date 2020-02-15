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




result,centroid = kmeans(train,3)
plot(result,centroid)

    
def Em

fig, ax = plt.subplots()
#result.plot.scatter(x = "X",y="Y",c = 'cluster',cmap = cmap,norm=norm)   
ax.scatter(result["X"],result["Y"],c = result['cluster'],cmap = cmap,s=20)
ax.scatter(centroid[0][0],centroid[0][1],marker = 'X',s = 100,label = 'centroid 1')
ax.scatter(centroid[1][0],centroid[1][1],marker = 'X',s = 100,label = 'centroid 2')
ax.scatter(centroid[2][0],centroid[2][1],marker = 'X',s = 100,label = 'centroid 3')
ax.legend(('X')('centroid'))

#fig.colorbar(ax.collections[0], ticks=np.linspace(0,2,3))

