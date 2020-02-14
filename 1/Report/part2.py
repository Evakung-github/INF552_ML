#!/usr/bin/env python
# coding: utf-8

#Group member
#Chenqi Liu   2082-6026-02
#Che-Pai Kung 5999-9612-95 
#Mengyu Zhang 3364-2309-80



import pandas as pd
import sklearn
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


col_names = ['Index', 'Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy']
# Load dataset
df = pd.read_csv('dt_data.txt',sep =': |, ', header=None, names=col_names)
df['Enjoy'] = df.Enjoy.apply(lambda x:x.strip(';'))
df.drop('Index',axis=1,inplace = True)


# Split dataset into attributes and label
attributes = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer']
X = df[attributes]
y = df.Enjoy

# Encode
X_encode = X.apply(preprocessing.LabelEncoder().fit_transform)
y_encode = y.map(dict(Yes=1, No=0))

# Mapping of text and code to visulize encoding
df_join = pd.concat([X, X_encode], axis=1)
occupied_map = df_join.Occupied.drop_duplicates()
price_map = df_join.Price.drop_duplicates()
music_map = df_join.Music.drop_duplicates()
location_map = df_join.Location.drop_duplicates()
vip_map = df_join.VIP.drop_duplicates()
beer_map = df_join['Favorite Beer'].drop_duplicates()

# Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X_encode, y_encode, test_size=0.25, random_state=42)


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Visualization
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = attributes,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decisiontree.png')
Image(graph.create_png())
