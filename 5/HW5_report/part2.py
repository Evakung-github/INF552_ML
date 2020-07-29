#!/usr/bin/env python
# coding: utf-8


"""
 Chenqi Liu 2082-6026-02 
 Che-Pai Kung 5999-9612-95 
 Mengyu Zhang 3364-2309-80
"""

from sklearn.neural_network import MLPClassifier
from readData import create_Data

trainX,trainY = create_Data("downgesture_train.list")
testX,testY = create_Data("downgesture_test.list")
trainY = trainY.ravel()

###########################################
# using fixed parameter
clf = MLPClassifier(solver='sgd',activation='logistic', hidden_layer_sizes=(100, 1), max_iter=1000, learning_rate_init=0.1, batch_size=10, tol=0.0000001)
clf.fit(trainX, trainY)
print(clf.predict(testX))
print(clf.score(testX, testY))

###########################################
# using adjusted parameters and best answer
clf = MLPClassifier(hidden_layer_sizes=(100, 1), max_iter=1000, activation='logistic')
clf.fit(trainX, trainY)

parameter_space = {
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05, 0.00005],
    'learning_rate': ['constant','adaptive'],
    'learning_rate_init': [0.001, 0.1],
    'tol': [1e-4, 1e-7]
}

from sklearn.model_selection import GridSearchCV

impr_clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)
impr_clf.fit(trainX, trainY)
print(impr_clf.predict(testX))
print(impr_clf.score(testX, testY))