# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:43:12 2020

@author: eva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from sklearn import cluster
from sklearn import mixture
from scipy.stats import multivariate_normal

data = pd.read_csv("clusters.txt",header = None)
data.columns = ["X","Y"]

KMeans = cluster.KMeans(n_clusters=3,random_state = 0).fit(data[['X','Y']])
KMeans.labels_
KMeans.predict([[2,2]])
KMeans.cluster_centers_

KMeans.transform([[2,2],[10,10]])   #計算到center的距離

data['cluster'] = kmeans.labels_

fig, ax = plt.subplots()   
ax.scatter(data["X"],data["Y"],c = data['cluster'],s=20)
ax.scatter(data["X"],data["Y"],c = data['EM'],s=20,cmap=cmap)

ax.scatter(result["X"],result["Y"],c = result['cluster'],s=20)


sum(result.cluster == data.cluster)



EM = mixture.GaussianMixture(n_components=3)
EM.fit(data[['X','Y']])
EM.predict(data[['X','Y']])
EM.get_params
proba = EM.predict_proba(data[['X','Y']])

proba.sum(axis=1)



data['EM'] = EM.predict(data[['X','Y']])

