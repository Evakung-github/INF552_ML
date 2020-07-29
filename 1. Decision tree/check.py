# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:00:58 2020

@author: eva
"""
#!pip install pydotplus
#check

from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import matplotlib.pyplot as plt
import graphviz
import numpy as np
import pandas as pd
#!pip install graphviz

train = pd.read_csv('dt_data.txt',sep =': |, ',engine = 'python')
train.columns = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer','Label']
train['Label'] = train.Label.apply(lambda x:x.strip(';'))
train.reset_index(inplace = True)
train.drop('index',axis=1,inplace = True)


test = pd.DataFrame([['Moderate','Cheap','Loud','City-Center','No','No'],\
                     ['High','Expensive','Loud','City-Center','Yes','Yes'],\
                     ['High','Normal','Quiet','German-Colony','No','Yes']],columns = train.columns[:-1])


library_train_X = train.iloc[:,:-1]
#test = pd.DataFrame([['Moderate','Cheap','Loud','City-Center','No','No']],columns = library_train_X.columns)
library_train = pd.concat([library_train_X,test])
library_train = library_train.apply(preprocessing.LabelEncoder().fit_transform)
library_train_X = library_train.iloc[:-3,:]
library_test = library_train.iloc[-3:,:]
le = preprocessing.LabelEncoder()
le.fit(train.iloc[:,-1])
library_train_Y = le.transform(train.iloc[:,-1])



dt = tree.DecisionTreeClassifier(criterion = "entropy")
dt.fit(library_train_X,library_train_Y)

dt.predict(np.array(library_test))

le.inverse_transform(dt.predict(np.array(library_test)))



#######


#
#
#
#dot_data = StringIO()
#export_graphviz(dt, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True,feature_names = library_train_X.columns,class_names=['0','1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.show()
#plt.show(graph)
#
#
#
#Image(graph.create_png())
#
#
#
#tree.export_graphviz(dt)
#pow(2,31)
#69*256**3+171*256**2+230*256+68
