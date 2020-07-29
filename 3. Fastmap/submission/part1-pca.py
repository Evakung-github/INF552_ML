#!/usr/bin/env python
# coding: utf-8
# Chenqi Liu 2082-6026-02 
# Che-Pai Kung 5999-9612-95 
# Mengyu Zhang 3364-2309-80

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = np.loadtxt("pca-data.txt")

x_array = df[:, 0]
y_array = df[:, 1]
z_array = df[:, 2]


# step 1: change avg to 0 - decentralization
x_mean = np.mean(x_array)
y_mean = np.mean(y_array)
z_mean = np.mean(z_array)

x_array -= x_mean
y_array -= y_mean
z_array -= z_mean


# step 2: get the covariance matrix
df_decen = df
df_decen[:,0] = x_array
df_decen[:,1] = y_array
df_decen[:,2] = z_array

df_decen_trans = df_decen.transpose()

cov_Matrix = df_decen_trans.dot(df_decen)


# step 3: compute eigenvalues and eigenvectors of covariance matrix
w, v = np.linalg.eig(cov_Matrix)


# step 4: choose the larget k eigenvalues w corresponding eigenvectors v
mapping = dict()
for i in range(len(w)):
    mapping[w[i]] = v[:, i]

w_sorted = sorted(w, reverse=True)
v_sorted = np.concatenate((mapping[w_sorted[0]], mapping[w_sorted[1]])).reshape(2,3)

v_sorted_trans = v_sorted.transpose()


# get final result
final = df_decen.dot(v_sorted_trans)


# save point to file
File = open("pca_output.txt","w")
for l in final:
    File.writelines(str(l[0])+" "+str(l[1])+"\n")
File.close()

# plot 2d
fig,ax = plt.subplots()
ax.scatter(final[:,0],final[:,1])
fig.set_size_inches(18.5, 10.5)
fig.savefig('part1-pca-2d.png', dpi=100)

# plot 3d
fig3d = plt.figure()
ax = fig3d.add_subplot(111, projection='3d')
ax.scatter(df_decen[:,0], df_decen[:,1], df_decen[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig3d.set_size_inches(18.5, 10.5, 10.5)
plt.show()
fig3d.savefig('part1-pca-3d.png', dpi=100)


