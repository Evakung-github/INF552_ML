# -*- coding: utf-8 -*-
"""
 Chenqi Liu 2082-6026-02 
 Che-Pai Kung 5999-9612-95 
 Mengyu Zhang 3364-2309-80
"""

import numpy as np
import matplotlib.pyplot as plt


def read_pgm(pgmf):
   """Return a raster of integers from a PGM as a list of lists."""
   assert pgmf.readline() == b'P5\n'
   """Comment line"""
   pgmf.readline()
   """Get size of the image"""
   (width, height) = [int(i) for i in pgmf.readline().split()]
   maxval = int(pgmf.readline())
   """
   Each gray value is represented in pure binary by either 1 or 2 bytes. 
   If the Maxval is less than 256, it is 1 byte. 
   Otherwise, it is 2 bytes.
   """
   assert maxval < 256

   raster = []
   for h in range(height):
       for w in range(width):
           raster.append(ord(pgmf.read(1)))
   return raster



def create_Data(file):
    file_list = []
    with open(file,'rb') as k:
        for line in k.readlines():
            file_list.append(line.strip())
    X = []
    Y = []
    for i in file_list:
        if b'down' in i:
            Y.append([1])
        else:
            Y.append([0])
        p = open(i,'rb')
        X.append(read_pgm(p))
    
    return [np.array(X),np.array(Y)]

trainX,trainY = create_Data("downgesture_train.list")
testX,testY = create_Data("downgesture_test.list")
