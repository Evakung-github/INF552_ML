# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:04:25 2020

@author: eva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("clusters.txt",header = None)
data.columns = ["X","Y"]

data.plot.scatter(x = "X",y="Y")