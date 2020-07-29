# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:12:35 2020

"""

#Group member
#Chenqi Liu   2082-6026-02
#Che-Pai Kung 5999-9612-95 
#Mengyu Zhang 3364-2309-80




#ref
#https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
#https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
#https://www.youtube.com/watch?v=LDRbO9a6XPU
#https://www.geeksforgeeks.org/decision-tree/


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Import train data
train = pd.read_csv('dt_data.txt',sep =': |, ',engine = 'python')
train.columns = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer','Label']
train['Label'] = train.Label.apply(lambda x:x.strip(';'))
train.reset_index(inplace = True,drop = True)

#Create test data
test = pd.DataFrame([['Moderate','Cheap','Loud','City-Center','No','No'],\
                     ['High','Expensive','Loud','City-Center','Yes','Yes'],\
                     ['High','Normal','Quiet','German-Colony','No','Yes']],columns = train.columns[:-1])

#Data structure used to build tree
class dt:
    def __init__(self,attribute):
        self.attribute = attribute
        self.true = None
        self.false = None
        
    
    
#Calculate the least entropy given column.
        
def column_least_entropy(df_X,columns,df_Y):
    #Find unique values of that column, so that we know what to split later.
    value = df_X[columns].unique()
    gain = 1.1
    condition=None
    groups= None
    
    for i in value: #Loop through each condition.
        true,false = df_X[df_X[columns] == i].index.values,df_X[df_X[columns] != i].index.values
        #If the value doesn't help splitting the data, then skip it.
        if len(true) ==0 or len(false)==0:
            continue
        
        #Calculate the entropy of true cases and false cases. 
        p_true = sum(df_Y.loc[true] == 'Yes')/len(true)
        p_false = sum(df_Y.loc[false] == 'Yes')/len(false)
        if p_true*(1-p_true)==0:
            entropy_t = 0
        else:
            entropy_t = (-p_true*np.log2(p_true)-(1-p_true)*np.log2(1-p_true))*len(true)/len(df_X)
        
        if p_false*(1-p_false) == 0:
            entropy_f = 0
        else:
            entropy_f = (-p_false*np.log2(p_false)- (1-p_false)*np.log2(1-p_false)) *len(false)/len(df_X)
        
        entropy = entropy_t+entropy_f
        
        #Document the condition and splitted groups if entropy is smaller.
        if entropy<gain:
            condition,gain = i,entropy
            groups = [true,false]
        
        #If there are only two values, then it doesn't need to calculate another one as they have same results. 
        if len(value) == 2:
            break
    if gain == 1.1:
        # It means that the data in df_X has the same value in column 
        # and it can't be further splitted based on this column.
        return gain,_,_
    else:
        return gain,condition,groups


def build_tree(df_X,df_Y):
    min_entropy =1.1
    
    if df_Y.nunique()==1:
        # The data is pure now.
        return df_Y.unique()[0]
    
    
    for col in df_X.columns:
        entropy,condition,groups = column_least_entropy(df_X,col,df_Y)
        if entropy == 0:
            # If entropy is 0, it leads to two pure subtrees, which is the best condition and lowest entropy.
            tree = dt([col,condition])
            tree.true = build_tree(df_X.loc[groups[0]],df_Y.loc[groups[0]])
            tree.false = build_tree(df_X.loc[groups[1]],df_Y.loc[groups[1]])
            return tree
            break
        if entropy<min_entropy:
            min_col = col
            min_condition = condition
            min_group = groups
            min_entropy = entropy
    if min_entropy == 1.1:
        # All attributes have the same values, but different labels.
        # Return the majority of labels as result.
        # If the number of each label are equal, then return 'half' and randomly choose one when predicting.
        t = sum(df_Y == 'Yes')
        f = sum(df_Y == 'No')
        if t==f:
            return 'half'
        elif t>f:
            return "Yes"
        else:
            return "No"
    # Build tree
    tree = dt([min_col,min_condition])
    tree.true = build_tree(df_X.loc[min_group[0]],df_Y.loc[min_group[0]])
    tree.false = build_tree(df_X.loc[min_group[1]],df_Y.loc[min_group[1]])
    return tree
            



def print_dt(tree,spacing=''):
    if isinstance(tree,str):
        print(spacing,"Predict-->",tree)
        return
    
    print(spacing,tree.attribute)
    
    print(spacing,"--TRUE:")
    print_dt(tree.true,spacing+"  ")
    
    print(spacing,"--FALSE:")
    print_dt(tree.false,spacing+"  ")



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
        if t == 'half':
            t = np.random.choice(['Yes','No'],1,[0.5,0.5])[0]
        
        ans.append(t)
        t = tree
    return pd.Series(ans)
     




X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:-1], train.iloc[:,-1], test_size=0.2, random_state=43)

t = build_tree(X_train,y_train)
t.attribute
print_dt(t)

X_test.reset_index(drop=True,inplace = True)
y_test.reset_index(drop=True,inplace = True)

y_predict = predict(t,X_test)

sum(y_predict == y_test)/len(y_test)

predict(t,test)
