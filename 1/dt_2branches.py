# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:12:35 2020

@author: eva
"""
#https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
#https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
#https://www.youtube.com/watch?v=LDRbO9a6XPU


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


train = pd.read_csv('dt_data.txt',sep =': |, ',engine = 'python')
train.columns = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer','Label']
train['Label'] = train.Label.apply(lambda x:x.strip(';'))
train.reset_index(inplace = True)
train.drop('index',axis=1,inplace = True)

train[train['Label'] == 'No'].index.values.tolist()
train['Label'].unique()[0]
#def Iscondition(df,columns,condition):
#    return [df[df[columns] == condition].index.tolist(),df[df[columns] != condition].index.tolist()]
test = pd.DataFrame([['Moderate','Cheap','Loud','City-Center','No','No'],\
                     ['High','Expensive','Loud','City-Center','Yes','Yes'],\
                     ['High','Normal','Quiet','German-Colony','No','Yes']],columns = train.columns[:-1])

class dt:
    def __init__(self,attribute):
        self.attribute = attribute
        self.true = None
        self.false = None


def information_gain(df,columns):
    value = df[columns].unique()
    gain = 1.1
    condition=None
    groups= None
    
    for i in value:
        true,false = df[df[columns] == i].index.values,df[df[columns] != i].index.values
        if len(true) ==0 or len(false)==0:
            continue
        
        p_true = sum(df.loc[true]['Label'] == 'Yes')/len(true)
        p_false = sum(df.loc[false]['Label'] == 'Yes')/len(false)
        if p_true*(1-p_true)==0:
            ig_t = 0
        else:
            ig_t = (-p_true*np.log2(p_true)-(1-p_true)*np.log2(1-p_true))*len(true)/len(df)
        
        if p_false*(1-p_false) == 0:
            ig_f = 0
        else:
            ig_f = (-p_false*np.log2(p_false)- (1-p_false)*np.log2(1-p_false)) *len(false)/len(df)
        
        info_gain = ig_t+ig_f
        
        if info_gain<=gain:
            condition,gain = i,info_gain
            groups = [true,false]
        if len(value) == 2:
            break
    if gain == 1.1:
        return gain,_,_
    else:
        return gain,condition,groups

information_gain(train,'Occupied')

def build_tree(df):
    min_info_gain =1.1
    
    if df['Label'].nunique()==1:
        return df['Label'].unique()[0]
    
    
    for col in df.columns[:-1]:
        info_gain,condition,groups = information_gain(df,col)
        if info_gain == 0:
            tree = dt([col,condition])
            tree.true = build_tree(df.loc[groups[0]])
            tree.false = build_tree(df.loc[groups[1]])
            return tree
            break
        if info_gain<=min_info_gain:
            min_col = col
            min_condition = condition
            min_group = groups
            min_info_gain = info_gain
    if min_info_gain == 1.1:
        t = sum(df['Label'] == 'Yes')
        f = sum(df['Label'] == 'No')
        if t==f:
            return 'half'
        elif t>f:
            return "Yes"
        else:
            return "No"
    tree = dt([min_col,min_condition])
    tree.true = build_tree(df.loc[min_group[0]])
    tree.false = build_tree(df.loc[min_group[1]])
    return tree
            

X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:-1], train.iloc[:,-1], test_size=0.2, random_state=42)
train_split = pd.concat([X_train,y_train],axis=1)




t = build_tree(train_split)
#tree= build_tree(train.loc[[6,9,11,17,20,21]])   
        
build_tree(train.loc[[10,19]])

#Print tree

#i = 0
#
#queue = [t]
#next_level = []
#while queue:
#    node = queue.pop(0)
#    if isinstance(node,str):
#        print(i,node,end =" ")
#        continue
#    else:
#        print(i,node.attribute,end =" ")
#    if node.true:
#        next_level.append(node.true)
#    if node.false:
#        next_level.append(node.false)
#    if not queue:
#        queue = next_level
#        next_level=[]
#        print()
#        i+=1


def print_dt(tree,spacing=''):
    if isinstance(tree,str):
        print(spacing,"Predict-->",tree)
        return
    
    print(spacing,tree.attribute)
    
    print(spacing,"--TRUE:")
    print_dt(tree.true,spacing+"  ")
    
    print(spacing,"--FALSE:")
    print_dt(tree.false,spacing+"  ")

print_dt(t)

t.attribute
train['Label'][0]
def predict(tree,test_data):
    t = tree
    ans = []
    for i in range(len(test_data)):
        while not isinstance(t,str):
            attribute,condition = t.attribute
            if test_data[attribute][i] == condition:
                t = t.true
            else:
                t = t.false
        ans.append(t)
        t = tree
    return pd.Series(ans)#,columns = ['Label'])
            
predict(t,X_test.loc[0])



(predict(t,train) == train['Label'])
    
type(train['Label'])
type(predict(t,train))
    

