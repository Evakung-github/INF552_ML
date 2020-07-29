# -*- coding: utf-8 -*-
"""
 Chenqi Liu 2082-6026-02 
 Che-Pai Kung 5999-9612-95 
 Mengyu Zhang 3364-2309-80
"""

import numpy as np
import matplotlib.pyplot as plt
from readData import create_Data
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

class layer:
    def __init__(self,N,prev,cur):
        self.old_weight = np.random.uniform(low = -1,high = 1,size = (prev,cur))*0.01
        self.weight = np.random.uniform(low = -1,high = 1,size = (prev,cur))*0.01
        self.s = np.zeros((N,cur))
        self.z = np.zeros((N,cur))
        self.diff = np.zeros((N,cur))
        self.cache = np.zeros_like(self.weight)+1e-4
          

    def activationFunction(self,s):
        return np.exp(s)/(1+np.exp(s))
    
    def diffOfactFunc(self,s):
        a = self.activationFunction(s)
        return a*(1-a)


class NN:
    def __init__(self,data,label,grad_opt = 'vanilla'):
        self.N = data.shape[0]
        self.Xtrain = data
        self.Xtrain = np.hstack((self.Xtrain,np.ones((self.N,1))))
        self.label = label
        
        self.dim = self.Xtrain.shape[1]
        self.structure = []
        self.size = [self.dim]

        self.grad_opt = grad_opt
    
    def addLayer(self,size):
        
        self.structure.append(layer(self.N,self.size[-1],size))
        self.size.append(size)
    
    def lossFunc(self):
        # return -(self.label*np.log(self.structure[-1].z)+(1-self.label)*np.log(1-self.structure[-1].z)).mean()
        return ((self.structure[-1].z-self.label)**2).mean()

    def diffOflossFunc(self,xo):
        diff = 2*(xo-self.label)/self.N
        # diff = -(self.label/self.structure[-1].z-(1-self.label)/(1-self.structure[-1].z))/self.N
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
            return

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
            self.structure[l].cache += (temp/len(b))**2
            if self.grad_opt == 'vanilla':
              if self.time_decay:
                self.structure[l].weight -= self.eta/np.sqrt(self.epoch+1)*(temp/len(b))
              else:
                self.structure[l].weight -= self.eta*(temp/len(b))
            elif self.grad_opt == 'Adagrad':
              self.structure[l].cache += (temp/len(b))**2
              self.structure[l].weight -= self.eta*(temp/len(b))/np.sqrt(self.structure[l].cache)
    
    
    
    def fit(self,eta = 0.1,max_epoch = 1000,tol = 0.0000001,batch = 10,time_decay = False):
        self.time_decay = time_decay
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
                self.BackPropagation(s[i*batch:min(self.N,(i+1)*batch)])
                # print("batch: {}, loss: {}".format(s[i*batch:min(self.N,(i+1)*batch)],l))
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

scaler = StandardScaler()
scaler.fit(trainX)

scale_trainX = scaler.transform(trainX)
scale_testX = scaler.transform(testX)

######################################################
model_ada = NN(scale_trainX,trainY,grad_opt='Adagrad')

model_ada.addLayer(100)
# model.addLayer(100)    

model_ada.addLayer(1)
model_ada.fit(max_epoch=1000,eta = 0.1,batch = 5)

ada_trainresult = model_ada.predict(scale_trainX,trainY)
ada_testresult = model_ada.predict(scale_testX,testY)

####################################################
model_vanilla = NN(trainX,trainY)

model_vanilla.addLayer(100)
# model.addLayer(100)

model_vanilla.addLayer(1)
model_vanilla.fit(max_epoch=1000,eta = 0.1,batch = 5)

vanilla_trainresult = model_vanilla.predict(trainX,trainY)
vanilla_testresult = model_vanilla.predict(testX,testY)

####################################################

print("train_nondown_gesture percentage:",1-trainY.sum()/len(trainY))
print("test_nondown_gesture percentage:",1-testY.sum()/len(testY))

#print(result[1])
#print(testresult[1])

#https://stackoverflow.com/questions/29266966/show-tick-labels-when-sharing-an-axis-in-matplotlib
fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2,
                        sharex=True,sharey = True,figsize=(14, 10))
ax0.set_title('Adagrad',fontdict={'fontsize':20})
ax0.plot(range(len(model_ada.losslist)),np.array(model_ada.losslist),lw = 1)
ax0.set_xlabel('epoch',fontsize = 14)
ax0.set_ylabel('loss',fontsize = 14)
ax0.annotate("{:.4f}".format(model_ada.losslist[-1]),(920,model_ada.losslist[-1]+0.01))
ax0.annotate("Accuracy of training data:{:.2f}%".format(ada_trainresult[1]*100),(200,0.25),size = 14)
ax0.annotate("Accuracy of testing data:{:.2f}%".format(ada_testresult[1]*100),(200,0.23),size = 14)
    

ax1.set_title('Mini-batch',fontdict={'fontsize':20})
ax1.plot(range(len(model_vanilla.losslist)),np.array(model_vanilla.losslist),lw = 1)
ax1.set_xlabel('epoch',fontsize = 14)
ax1.set_ylabel('loss',fontsize = 14)
ax1.yaxis.set_tick_params(labelbottom=True)
ax1.annotate("{:.4f}".format(model_vanilla.losslist[-1]),(920,model_ada.losslist[-1]+0.01))
ax1.annotate("Accuracy of training data:{:.2f}%".format(vanilla_trainresult[1]*100),(200,0.25),size = 14)
ax1.annotate("Accuracy of testing data:{:.2f}%".format(vanilla_testresult[1]*100),(200,0.23),size = 14)

fig.suptitle('Loss after each epoch',fontsize = 24)

plt.show()