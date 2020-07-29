# group member
# Chenqi Liu 2082-6026-02 
# Che-Pai Kung 5999-9612-95 
# Mengyu Zhang 3364-2309-80


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv("pca-data.txt", sep="	", header=None)
df.columns = ["x", "y", "z"]
dataset = df.astype(float).values.tolist()
X = df.values #returns a numpy array

pca = PCA(n_components=2)
pca.fit(X)

X_pca = pca.transform(X)

wf = pd.DataFrame(X_pca)
wf.to_csv("pca-2d.txt", index = False)