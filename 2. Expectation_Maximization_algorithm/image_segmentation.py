# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:31:16 2020

@author: eva
"""

#!pip install opencv-python
#https://towardsdatascience.com/introduction-to-image-segmentation-with-k-means-clustering-83fd0a9e2fc3

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image 

original_image = cv2.imread("pic1.jpeg")

cv2.imshow('image',original_image)

ori = original_image.reshape((-1,3))
ori = np.float32(ori)


img = Image.open('pic1.jpeg')
img.show()
