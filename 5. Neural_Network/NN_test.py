# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:32:40 2020

@author: eva
"""

import numpy as np
import matplotlib.pyplot as plt

class layer:
    def __init__(self,N,prev,cur):
#        if cur > 1:
        self.old_weight = np.random.uniform(low = -1,high = 1,size = (prev,cur))*0.01
        self.weight = np.random.uniform(low = -1,high = 1,size = (prev,cur))*0.01
#        self.weight = np.random.randint(low = 5,size = (prev,cur))
        self.s = np.zeros((N,cur))
        self.z = np.zeros((N,cur))
        self.diff = np.zeros((N,cur))
#        else:
#            self.old_weight = np.random.uniform(low = -1,high = 1,size = (prev,))*0.01
#            self.weight = np.random.uniform(low = -1,high = 1,size = (prev,))*0.01
#            self.s = np.zeros((N,))
#            self.z = np.zeros((N,))
#            self.diff = np.zeros((N,))           

    def activationFunction(self,s):
        return np.exp(s)/(1+np.exp(s))
    
    def diffOfactFunc(self,s):
        a = self.activationFunction(s)
        return a*(1-a)



class NN:
    def __init__(self,data,label):
        self.N = data.shape[0]
        self.Xtrain = data
        self.Xtrain = np.hstack((self.Xtrain,np.ones((self.N,1))))
        self.label = label
        
        self.dim = self.Xtrain.shape[1]
        self.structure = []
        self.size = [self.dim]
    
    def addLayer(self,size):
        
        self.structure.append(layer(self.N,self.size[-1],size))
        self.size.append(size)
    
    def loss(self):
        return ((self.structure[-1].z-self.label)**2).mean()

    def diffOflossFunc(self,xo):
        diff = 2*(xo-self.label)/self.N
        return diff
    
    def FeedForward(self):
        
        for i in range(len(self.structure)):
            
            if i ==0:
                self.structure[i].s = self.Xtrain @ self.structure[i].weight
                self.structure[i].z = self.structure[i].activationFunction(self.structure[i].s)
            else:
                self.structure[i].s = self.structure[i-1].z @ self.structure[i].weight
                self.structure[i].z = self.structure[i].activationFunction(self.structure[i].s)
                
    
    def BackPropagation(self):
        '''
        diff: array(N,)
        
        '''
        diff = self.diffOflossFunc(self.structure[-1].z)
        for l in range(len(self.structure)-1,-1,-1):
#            print(l)
            if l == len(self.structure) -1:
                self.structure[l].diff = self.structure[l].diffOfactFunc(self.structure[l].s)*diff
            else:
                self.structure[l].diff = self.structure[l].diffOfactFunc(self.structure[l].s)*(self.structure[l+1].diff@self.structure[l+1].old_weight.T)
            self.structure[l].old_weight = self.structure[l].weight
            temp = []
            i = np.random.randint(low = self.N)
            if l == 0:
#                for i in range(self.N):
                temp.append(self.Xtrain[i].reshape(-1,1)*self.structure[l].diff[i])
            else:
#                for i in range(self.N):
                temp.append(self.structure[l-1].z[i].reshape(-1,1)*self.structure[l].diff[i])
            self.structure[l].weight -= self.eta*np.array(temp).mean(axis = 0)
    
    
    
    def fit(self,eta = 0.1,max_epoch = 1000,tol = 0.000000001):
        self.eta = eta
        self.epoch = 0
        self.losslist = []
        prev_loss = float('inf')
        self.min_loss = float('inf')
        diff = -1
        while self.epoch<max_epoch and abs(diff)>tol:
            self.FeedForward()
            l = self.loss()
            self.losslist.append(l)
            diff,prev_loss = l - prev_loss,l
            self.BackPropagation()
            self.epoch+=1
        
        pass
    
    
    def predict(self,Xtest,ytest = []):
        Xtest = np.hstack((Xtest,np.ones((len(Xtest),1))))
        self.temp_result_s = []
        self.temp_result_z = []
        for i in range(len(self.structure)):
            
            if i ==0:
                self.temp_result_s.append(Xtest @ self.structure[i].weight)
                self.temp_result_z.append(self.structure[i].activationFunction(self.temp_result_s[-1]))
            else:
                self.temp_result_s.append(self.temp_result_z[-1] @ self.structure[i].weight)
                self.temp_result_z.append(self.structure[i].activationFunction(self.temp_result_s[-1]))
        if len(ytest)>0:
            accuracy = (np.where(self.temp_result_z[-1]-0.5<0,0,1) == ytest).sum()/len(Xtest)
            
        return [np.where(self.temp_result_z[-1]-0.5<0,0,1),accuracy if len(ytest)>0 else None]
        
    
    
            
trainY = trainY.reshape(-1,1)            
model = NN(trainX,trainY)

model.addLayer(1000)
model.addLayer(1)
model.fit(max_epoch=1000)
model.structure[0].z.shape

model.epoch
model.FeedForward()
model.BackPropagation()
(np.where(model.structure[-1].z-0.5<0,0,1) == trainY).sum()/model.N

a = model.predict(testX,testY)
b = np.where(model.structure[-1].z-0.5<0,0,1)
(np.where(model.temp_result_z[-1]-0.5<0,0,1) == testY).sum()/83

b == testY

np.equal(b,testY)
m = NN(np.array([[1,2]]),np.array([[0.01,0.99]]))

#m.Xtrain
m.addLayer(2)
#m.structure[-1].weight = np.array([[1.0,2.0],[3.0,4.0],[5.0,5.0]])
m.addLayer(2)
#m.structure[-1].weight = np.array([[1.0,2.0],[3.0,4.0]])

m.FeedForward()
m.eta = 0.1
m.BackPropagation()
m.fit()
m.epoch
m.N
m.structure[1].z
m.loss()
m.eta = 0.1

diff = m.diffOflossFunc(m.structure[-1].z)
diff = diff/2
l = 0
diff[0][1]*m.structure[1].z*(1-m.structure[1].z)
for l in range(len(self.structure)-1,-1,-1):
#            print(l)
    if l == len(self.structure) -1:
        m.structure[l].diff = m.structure[l].diffOfactFunc(m.structure[l].s)*diff
    else:
        m.structure[l].diff = m.structure[l].diffOfactFunc(m.structure[l].s)*(m.structure[l+1].diff@m.structure[l+1].old_weight.T)
    m.structure[l].old_weight = m.structure[l].weight
    m.structure[l].weight -= m.eta*(m.structure[l-1].z.T*m.structure[l].diff).reshape(-1,m.structure[l].weight.shape[1])

m.structure[l-1].z.T*m.structure[l].diff

(0.97**2)/2
fig, ax0 = plt.subplots()
ax0.set_title('loss(Ein)')
ax0.plot(range(len(model.losslist)),np.array(model.losslist),lw = 1)  

pred = model.structure[-1].z
((pred - trainY)**2).sum()/184

model.fit()
l = 1
model.structure[l].diff.shape@model.structure[l-1].z