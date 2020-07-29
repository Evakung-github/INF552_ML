# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:21:11 2020

Group Members:
Che-Pai Kung 5999-9612-95
Chenqi Liu 2082-6026-02 
Mengyu Zhang 3364-2309-80

'''


import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
import matplotlib as mpl
sns.set()
             
def linear_kernel(Xpos, Xneg):
    return np.dot(Xpos, Xneg)

def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p

class SVM(object):

    def __init__(self, kernel=linear_kernel):
        self.kernel = kernel
        # print(self.kernel)
        # print(linear_kernel)
        # print(polynomial_kernel)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix

        K = self.kernel(X,X.T)

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples),'d')
        b = cvxopt.matrix(0.0)

        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

        # solve QP problem using cvxopt solver
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers, we used threshold of 1e-4 here
        # anyvalue smaller than this will not be considered as support vectors
        sv = a > 1e-4
        ind = np.arange(len(a))[sv]

        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors were found" % (len(self.a)))
        
        print('Support Vectors are:')
        for i in self.sv:
          print(i)

        # Intercept
        i = ind[0]
        
        self.b = y[i] - np.sum(self.a * self.sv_y * K[sv,i])

        # Weight vector
        if self.kernel == linear_kernel:
            # print("?????")
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            # print("here?")
            self.w = None

        print("weight is ")
        print(self.w)
        print("intercept is: ")
        print(self.b)


    def project(self, X):
        return self.a.reshape(1,-1)*self.sv_y @self.kernel(self.sv,X.T)+self.b

    def predict(self, X):
        return np.sign(self.project(X))
    
    def plot_result(self,X,y,df = False):
        colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
        cmap = mpl.colors.ListedColormap(colors)
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", ['#7bc8f6','#af6f09'])

        
        plt.figure(1, figsize=(10, 6))
        ax = plt.gca()

        plt.scatter(X[:,0],X[:,1],c=y,cmap = plt.cm.Paired,s = 30)
        if df:
          
          xmin,xmax = ax.get_xlim()
          ymin,ymax = ax.get_ylim()

          xx = np.linspace(xmin,xmax,200)
          #yy = -(xx*w[0]+b)/w[1]
          yy = np.linspace(ymin,ymax,200)
          xx,yy = np.meshgrid(xx, yy)
          Z = self.project(np.c_[xx.ravel(), yy.ravel()])
          self.Z = Z

          #ax.plot(xx,yy)
          # plt.figure(1, figsize=(4, 3))
          ax.pcolormesh(xx,yy,Z.reshape(xx.shape)>0,cmap=cmap)
          # ax.set_xlim(x_lim)
          # ax.set_ylim(y_lim)
          plt.contour(xx, yy, Z.reshape(xx.shape), colors = 'k',levels=[-1, 0, 1], alpha=0.5,
                      linestyles=['--', '-', '--'])
          plt.scatter(X[y==1,0],X[y==1,1],c='#feb308',s = 30,label = 1)
          plt.scatter(X[y==-1,0],X[y==-1,1],c='#436bad',s = 30,label = -1)
          plt.scatter(self.sv[:, 0], self.sv[:, 1], s=100,
                      linewidth=1, facecolors='none', edgecolors='k',label = 'sv')
          
          ax.legend()
          ax.set_xlim((xmin,xmax))
          ax.set_ylim((xmin,xmax))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

if __name__ == "__main__":
    import pylab as pl

    def read_data():
        X_train = []
        Y_train = []
        with open("linsep.txt") as f:
            line = f.readline()
            while line:
                row = line.split(",")
                coord=[]
                coord.append(float(row[0]))
                coord.append(float(row[1]))
                l = row[-1]
                
                if l[0]=='+':
                    X_train.append(coord)
                    Y_train.append(1)
                else:
                    X_train.append(coord)
                    Y_train.append(-1)

                line = f.readline()

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        return X_train, Y_train

    def read_data_nonlin():
        X_train = []
        Y_train = []
        with open("nonlinsep.txt") as f:
            line = f.readline()
            while line:
                row = line.split(",")
                coord=[]
                coord.append(float(row[0]))
                coord.append(float(row[1]))
                l = row[-1]
                
                if l[0]=='+':
                    X_train.append(coord)
                    Y_train.append(1)
                else:
                    X_train.append(coord)
                    Y_train.append(-1)

                line = f.readline()

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        return X_train, Y_train


    def test_linear():
        X_train, y_train = read_data()
        qp = SVM()
        qp.fit(X_train, y_train)

        qp.plot_result(X_train,y_train,True)

    def test_non_linear():
        X_train, y_train = read_data_nonlin()
        qp = SVM(polynomial_kernel)
        qp.fit(X_train, y_train)

        qp.plot_result(X_train,y_train,True)

test_linear()
test_non_linear()