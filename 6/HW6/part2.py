# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:05:59 2020

Group Members:
Che-Pai Kung 5999-9612-95
Chenqi Liu 2082-6026-02 
Mengyu Zhang 3364-2309-80

"""

#https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py
#https://www.digitalvidya.com/blog/understanding-support-vector-machines-and-its-applications/
#https://chrisalbon.com/machine_learning/support_vector_machines/imbalanced_classes_in_svm/


from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

data = np.loadtxt('linsep.txt',delimiter = ',')
nonlin_data = np.loadtxt('nonlinsep.txt',delimiter = ',')
Xtrain = data[:,:2]
Ytrain = data[:,2]
nonlin_Xtrain = nonlin_data[:,:2]
nonlin_Ytrain = nonlin_data[:,2]


def svm_plot(X,Y,k,g='scale',c = 1,d = 2,coef0 = 0):
  colors = list(mpl.colors.TABLEAU_COLORS)

  clf = SVC(kernel=k, gamma=g,C = c,degree = d,coef0 = coef0)
  clf.fit(X, Y)

  # plot the line, the points, and the nearest vectors to the plane
  plt.figure(1, figsize=(10, 6))
  plt.clf()

  plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
              facecolors='none', zorder=10, edgecolors='k',label = 'Supported vectors')
  # plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
  #             edgecolors='k')
  
  plt.scatter(X[Y==1, 0], X[Y==1, 1], c=colors[1], zorder=10, label = '+1')
  plt.scatter(X[Y==-1, 0], X[Y==-1, 1], c=colors[0], zorder=10, label = '- 1')

  plt.axis('tight')
  ax = plt.gca()
  x_min,x_max = ax.get_xlim()
  y_min,y_max = ax.get_ylim()

  XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
  Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(XX.shape)
  plt.figure(1, figsize=(10, 6))
  plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
  plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
              levels=[-1, 0, 1])
  plt.legend()
  plt.title(k,size = 20)
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)

  plt.xticks(())
  plt.yticks(())
  plt.show()

  return clf

m = svm_plot(Xtrain,Ytrain,'linear',c = 1000)
print('linear sv: \n',m.support_vectors_)
print('linear score:',m.score(Xtrain,Ytrain))

m = svm_plot(nonlin_Xtrain,nonlin_Ytrain,'poly',c=1000)
print('poly sv: \n',m.support_vectors_)
print('poly score:',m.score(nonlin_Xtrain,nonlin_Ytrain))

#m = svm_plot(nonlin_Xtrain,nonlin_Ytrain,'rbf',c=1000)
#print('rbf sv: \n',m.support_vectors_)