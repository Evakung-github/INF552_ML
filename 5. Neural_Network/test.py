# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:21:30 2020

@author: eva
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
#   print(height,width,maxval)
#   for h in range(height):
#       row = []
#       for w in range(width):
#           a= pgmf.read(1)
#           low_bits = ord(a)
#           row.append(low_bits)
#       raster.append(row)
   for h in range(height):
       for w in range(width):
           raster.append(ord(pgmf.read(1)))
   return raster


file_list = []

with open("downgesture_train.list","rb") as k:
    for line in k.readlines():
        file_list.append(line.strip())


trainX = []
trainY = []
for i in file_list:
    if b'down' in i:
        trainY.append(1)
    else:
        trainY.append(0)
    p = open(i,'rb')
    trainX.append(read_pgm(p))
    

trainX = np.array(trainX)
trainY = np.array(trainY)

file_list = []

with open("downgesture_test.list","rb") as k:
    for line in k.readlines():
        file_list.append(line.strip())


testX = []
testY = []
for i in file_list:
    if b'down' in i:
        testY.append(1)
    else:
        testY.append(0)
    p = open(i,'rb')
    testX.append(read_pgm(p))
    

testX = np.array(testX)
testY = np.array(testY).reshape(-1,1)








plt.imshow(np.array(a),cmap='gray')


from PIL import Image

def read_img():
    im = Image.open("gestures/E/E_down_1.pgm")    # 读取文件
    im.show()    # 展示图片
    print(im.size)   # 输出图片大小
read_img()