# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:43:46 2020

@author: eva
"""

import numpy as np
import matplotlib.pyplot as plt
from readData import create_Data

class layer:
    def __init__(self,N,prev,cur):
        self.old_weight = np.random.uniform(low = -1,high = 1,size = (prev,cur))*0.01
        self.weight = np.random.uniform(low = -1,high = 1,size = (prev,cur))*0.01
        self.s = np.zeros((N,cur))
        self.z = np.zeros((N,cur))
        self.diff = np.zeros((N,cur))
          

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
    
    def lossFunc(self):
        return ((self.structure[-1].z-self.label)**2).mean()

    def diffOflossFunc(self,xo):
        diff = 2*(xo-self.label)/self.N
        return diff
    
    def FeedForward(self):       
        for i in range(len(self.structure)):           
            if i ==0:
                self.structure[i].s = self.Xtrain @ self.structure[i].weight
            else:
                self.structure[i].s = self.structure[i-1].z @ self.structure[i].weight
            self.structure[i].z = self.structure[i].activationFunction(self.structure[i].s)
    
    def BackPropagation(self,b):
        '''
        diff: array(N,self.label output)
        '''
        if len(b)==0:
            pass
#        if batch>self.N:
#            batch = self.N+1
#        b = np.random.randint(low = self.N,size = (batch))
        diff = self.diffOflossFunc(self.structure[-1].z)
        for l in range(len(self.structure)-1,-1,-1):
            if l == len(self.structure) -1:
                self.structure[l].diff = self.structure[l].diffOfactFunc(self.structure[l].s)*diff
            else:
                self.structure[l].diff = self.structure[l].diffOfactFunc(self.structure[l].s)*(self.structure[l+1].diff@self.structure[l+1].old_weight.T)
            self.structure[l].old_weight = self.structure[l].weight
            temp = np.zeros_like(self.structure[l].weight)

            
            if l == 0:
                for i in b:
                    temp += self.Xtrain[i].reshape(-1,1)*self.structure[l].diff[i]
            else:
                for i in b: 
                    temp += self.structure[l-1].z[i].reshape(-1,1)*self.structure[l].diff[i]
            self.structure[l].weight -= self.eta*temp/len(b)
    
    
    
    def fit(self,eta = 0.1,max_epoch = 1000,tol = 0.0000001,batch = 10):
        self.eta = eta
        self.epoch = 0
        self.losslist = []
        prev_loss = float('inf')
        self.min_loss = float('inf')
        diff = -1
        dc = 0
        while self.epoch<max_epoch and dc<10:
            self.FeedForward()
            l = self.lossFunc()
            self.losslist.append(l)
            diff,prev_loss = l - prev_loss,l
            s = np.arange(self.N)
            np.random.shuffle(s)

            for i in range(self.N//batch+1):
                self.BackPropagation(s[i:min(self.N,(i+1)*batch)])
            self.epoch+=1
            
            if self.epoch % 50 == 0:
                print("Epoch: {}, loss: {}".format(self.epoch,l))
            
            if abs(diff)<tol:
                dc+=1
            else:
                dc = 0
        
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
    

trainX,trainY = create_Data("downgesture_train.list")
testX,testY = create_Data("downgesture_test.list")

model = NN(trainX,trainY)

model.addLayer(100)
model.addLayer(100)
model.addLayer(1)
model.fit(max_epoch=1000,eta = 0.1,batch = 2)

model.epoch
1-trainY.sum()/len(trainY)
1-testY.sum()/len(testY)



result = model.predict(trainX,trainY)
testresult = model.predict(testX,testY)

fig, ax0 = plt.subplots()
ax0.set_title('loss(Ein)')
ax0.plot(range(len(model.losslist)),np.array(model.losslist),lw = 1)  






