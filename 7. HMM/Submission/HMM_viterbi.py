# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 20:11:41 2020

Group members:
Che-Pai Kung 5999-9612-95
Chenqi Liu 2082-6026-02 
Mengyu Zhang 3364-2309-80

"""
#http://www.csie.ntnu.edu.tw/~u91029/HiddenMarkovModel.html

#!pip install hmmlearn

import numpy as np
import matplotlib.pyplot as plt

#read data
grid_world = np.loadtxt('hmm-data.txt',skiprows=2,max_rows=10)
observations = np.loadtxt('hmm-data.txt',skiprows=24,max_rows=11)

grid_world.sum()

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
    return transition_matrix,initial_prob,distance

transition_matrix,initial_prob,distance = createMatrix()
#main program
def viterbi(initial_prob,transition_matrix,distance,observations):
    state_number = transition_matrix.shape[0]
    state_t = {}
    logprob_t = np.zeros((state_number,1))
    logprob_t.fill(-float('inf'))
    
    possible_state_t = np.where(np.logical_and(distance>= observations[0]/1.3, distance<= observations[0]/0.7).sum(axis = 1)==4)[0]
    for i in possible_state_t:
      logprob_t[i] = np.log(initial_prob[i])+np.log(1/(distance[i]*(1.3-0.7))).sum()
      state_t[i] = [i]
    
    for i in range(1,11):
    
      state_t_1 = state_t
      logprob_t_1 = logprob_t
      possible_state_t_1 = possible_state_t
    
    
      state_t = {}
      possible_state_t = np.where(np.logical_and(distance>= observations[i]/1.3, distance<= observations[i]/0.7).sum(axis = 1)==4)[0]
      logprob_t = np.zeros((state_number,1))
      logprob_t.fill(-float('inf'))
    
    
      for cur in possible_state_t:
        max_ = -float('inf')
        state = -1
        for prev in possible_state_t_1:
          if transition_matrix[prev][cur] == 0:
            continue
          c = (np.round(distance[cur]*1.3,1) - np.round(distance[cur]*0.7,1))/0.1+1
          p = logprob_t_1[prev] + np.log(transition_matrix[prev][cur])+np.log(1/c).sum()
          # if cur == 63:
          #   print(prev,p)
          if p > max_:
            max_ = p
            state = prev
        
        if state != -1:
          logprob_t[cur] = max_
          state_t[cur] = state_t_1[state]+[cur]
    
    return state_t[np.argmax(logprob_t)]
    
state_seq = viterbi(initial_prob,transition_matrix,distance,observations)

result = np.array(list(map(lambda x:[x//10,x%10],state_seq)))
print([(i[0],i[1]) for i in result])

N,M = len(grid_world),len(grid_world[0])

def plot(result):
    plt.figure(1, figsize=(10, 6))
    plt.scatter(result[:,1],result[:,0])
    for i in range(N):
      for j in range(M):
          if grid_world[i][j] == 0:
              plt.scatter(j,i,c = 'red',label = 'obstacles')
    count = 0
    for i in result:
      #plt.annotate(observations[count],(i[0]-1,i[1]+0.5))
      plt.annotate(count,(i[1]-0.1,i[0]+0.5))
      count+=1
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.invert_yaxis()
#    plt.xlabel('Column')
#    plt.ylabel('Row')
    plt.xlim((0,9))
    plt.ylim((9,0))
    plt.legend(['trajectory','obstacles'])
    plt.show()

plot(result)