# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:43:12 2020

@author: eva
"""
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?highlight=kmeans#sklearn.cluster.KMeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import cluster
from sklearn import mixture
from scipy.stats import multivariate_normal


#Load data
data = pd.read_csv("clusters.txt",header = None)
data.columns = ["X","Y"]

#plot function
def plot(result,centroid,title):
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
        plt.title(title)
        i+=1
    ax.legend([a],['mean'])
    plt.show()


#KMeans algorithm

KMeans = cluster.KMeans(n_clusters=3,init = "random",n_init = 10).fit(data[['X','Y']])
#KMeans.labels_                      #Result of KMeans
#KMeans.predict([[2,2]])             #Predict
#KMeans.cluster_centers_             #centroids of each clusger   
#KMeans.transform([[2,2],[10,10]])   #calculate the distance to the centroids

data['cluster'] = KMeans.labels_

plot(data,KMeans.cluster_centers_,"sklearn_cluster_kmeans")


#############################################################
#https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
#EM algorithm
cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen"])

EM = mixture.GaussianMixture(n_components=3,init_params='kmeans',n_init = 5,max_iter=500)
EM.fit(data[['X','Y']])
#EM.predict(data[['X','Y']])
##EM.get_params
#EM.means_                            # The mean of each mixture component.
#EM.covariances_                      # The covariance of each mixture component.
#EM.lower_bound_                      # Lower bound value on the log-likelihood (of the training data with respect to the model) of the best fit of EM.
#EM.n_init                            # Number of step used by the best fit of EM to reach the convergence. 
#EM.predict_proba(data[['X','Y']])


data['cluster'] = EM.predict(data[['X','Y']])
#Generate EM's centroids
centroids = np.array(data.groupby('cluster').mean())
plot(data,EM.means_,'sklearn_mixture_GauussianMixture')

#data.plot.scatter(x = "X",y="Y",c = 'cluster',cmap = cmap)