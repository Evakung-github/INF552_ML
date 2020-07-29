# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:19:03 2020

Group members:
Che-Pai Kung 5999-9612-95
Chenqi Liu 2082-6026-02 
Mengyu Zhang 3364-2309-80
"""

from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

#read data
grid_world = np.loadtxt('hmm-data.txt',skiprows=2,max_rows=10)
observations = np.loadtxt('hmm-data.txt',skiprows=24,max_rows=11)


#Create transition matrix, start probability and distance (for emission)
def createMatrix():
    N,M = len(grid_world),len(grid_world[0])
    transition_matrix = np.zeros((N*M,N*M))
    initial_prob = np.ones((N*M,1))
    distance = np.zeros((N*M,4))
    
    for i in range(N):
      for j in range(M):
        if grid_world[i][j] ==0:
          initial_prob[i*10+j]=0
          transition_matrix[i*10+j][i*10+j] = 1
          continue
        else:
          transition_matrix[i*10+j][max(i-1,0)*10+j] = grid_world[max(i-1,0)][j]
          transition_matrix[i*10+j][min(i+1,M-1)*10+j] = grid_world[min(i+1,M-1)][j]
          transition_matrix[i*10+j][i*10+max(j-1,0)] = grid_world[i][max(j-1,0)]
          transition_matrix[i*10+j][i*10+min(j+1,N-1)] = grid_world[i][min(j+1,N-1)]
          transition_matrix[i*10+j][i*10+j] = 0
          for c,t in enumerate([[0,0],[0,9],[9,0],[9,9]]):
            distance[i*10+j][c] = np.sqrt((i-t[0])**2+(j-t[1])**2)
            
    transition_matrix = transition_matrix/np.where(transition_matrix.sum(axis=0)==0,1,transition_matrix.sum(axis=0)).reshape(-1,1)
    initial_prob = initial_prob/initial_prob.sum()
    emissionprob = np.zeros((N*M,len(observations)))
    for i in range(len(observations)):
      k = np.where(np.logical_and(distance>= observations[i]/1.3, distance<= observations[i]/0.7).sum(axis = 1)==4)[0]
      for j in k:
        emissionprob[j][i] = np.prod(1/(distance[j]*(1.3-0.7)))
    
    
    
    return transition_matrix,initial_prob,distance,emissionprob

transition_matrix,initial_prob,distance,emissionprob = createMatrix()


    
###Main program
m = hmm.MultinomialHMM(n_components=100,params='',init_params='s')

m.transmat_ = transition_matrix
m.startprob = initial_prob
m.emissionprob_ = emissionprob

m.fit(np.array(range(11)).reshape(-1,1))

m.startprob_

print(m.decode(np.arange(11).reshape(-1,1)))
#[53, 63, 73, 83, 82, 72, 71, 61, 51, 41, 31]

