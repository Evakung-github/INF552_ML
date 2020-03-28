# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:58:17 2020

@author: eva
"""

import sklearn.preprocessing as sp

d = sp.scale(data)
d.sum(axis = 0)
d[:,:2].shape
d = pd.DataFrame(d,columns=['X','Y','cluster'])
KM = cluster.KMeans(n_clusters=3,init = "random").fit(d[:,:2])

d['cluster'] = KM.labels_
plot(d,KM.cluster_centers_)



EM = mixture.GaussianMixture(n_components=3,n_init =5,init_params='random')
EM.fit(d[['X','Y']])


d['cluster'] = EM.predict(d[['X','Y']])
#Generate EM's centroids
centroids = np.array(d.groupby('cluster').mean())
plot(d,EM.means_)