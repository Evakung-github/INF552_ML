# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:43:24 2020

@author: eva
"""

import numpy as np
import pandas as pd

train = pd.read_csv('dt_data.txt',sep =': |, ',engine = 'python')
train.columns = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer','Label']
train['Label'] = train.Label.apply(lambda x:x.strip(';'))
train.reset_index(inplace = True)
train.drop('index',axis=1,inplace = True)

train.Label.nunique()

train.describe()
a = [1,5,6]
train.iloc[a]
list(map(lambda x:x[1].index.values,train.groupby('VIP')))[0]

# wrong:train.groupby('VIP').apply(lambda x:x[1].index)

def dt(df):
    cur = [list(range(len(df)))]
    next_level = []
    attributes=[]
    while cur:
        node = cur.pop(0)
        if len(node)>0:
            min_info_gain =1
            groups =[]
            total = len(node)
            if df.iloc[node]['Label'].nunique()==1:
                groups = None
                attribute = None
                print(groups)
                continue
            for i in df.columns[:-1]:
                temp = list(map(lambda x:x[1].index.values,df.iloc[node].groupby(i)))
                info_gain =0
                for j in temp:
                    p = sum(df.iloc[j]['Label'] == 'Yes')/len(j)
                    if p ==0 or 1-p ==0:
                        info_gain+=0
                    else:
                        info_gain += (-p*np.log2(p)-(1-p)*np.log2(1-p))*len(j)/total
                if info_gain == 0:
                    groups = temp
                    attribute =i
                    break
                elif info_gain<min_info_gain:
                    min_info_gain=info_gain
                    groups = temp
                    attribute =i
            print(attribute,groups)

            next_level.extend(groups)
            attributes.append(attribute)
            if not cur:
                cur = next_level
                next_level=[]
            
        else:
            next_level.append(None)
            attributes.append(None)

    return attributes

dt(train)



min_info_gain = 1
groups = []
total = len(train)
for i in train.iloc[:,:-1].columns:
    temp = list(train.groupby(i))
    info_gain = 0
    for j in temp:
        p = sum(j[1]['Label'] =='Yes')/len(j[1])
        if p == 0 or 1-p ==0:
            info_gain += 0
        else:
            info_gain += (-p*np.log2(p)-(1-p)*np.log2(1-p))*len(j[1])/total
    
    if info_gain<min_info_gain:
        min_info_gain = info_gain
        groups = temp
        attributes = i