from sklearn import model_selection
from sklearn import linear_model
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn import naive_bayes, svm, tree
from sklearn.neural_network import MLPClassifier
import numpy as np
from time import *
from sklearn.model_selection import GridSearchCV


df=pd.read_csv('train_washed.csv')
x = df.values[:,2:]
y = df.values[:,1]
# # LR
# clf = linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-6)
# param_dist = [
#         {
#         'C':np.arange(0.1,1,0.01)
#          }
# ]
# ada
# clf = ada(linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-6), algorithm='SAMME.R',
#           learning_rate=0.7, n_estimators=65, random_state=7)
# param_dist = [
#         {
#         'learning_rate':np.arange(0.1,1,0.01),
#         'n_estimators':np.arange(1,100,1)
#          }
# ]
# SVM
# clf = svm.SVC(C=0.7,kernel='rbf',random_state=7)
# param_dist = [
#         {
#         'C':np.arange(0.1,1,0.01),
#         'kernel':['linear', 'poly', 'rbf', 'sigmoid']
#          }
# ]
# Bagging
# clf = BaggingClassifier(linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-6),
#                        n_estimators=60,max_samples=0.5, max_features=0.5, n_jobs=-1)
# param_dist = [
#         {
#         'n_estimators':np.arange(1,100,1)
#          }
# ]
# DT
# clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=1,min_samples_split=2,
#                                   random_state=7)
# param_dist = [
#         {
#         'criterion':['gini','entropy'],
#         'max_depth':np.arange(1,10,1),
#         'min_samples_leaf':np.arange(1,10,1),
#         'min_samples_split':np.arange(2,10,1)
#          }
# ]
# NN
clf = MLPClassifier(activation='relu',hidden_layer_sizes=(10,2),random_state=7,solver='adam')
hidden_layer=[]
for i in range(15,40):
    # for j in range(10):
        hidden_layer.append((i,2))
param_dist = [
        {
        # 'activation':['tanh','logistic','relu'],
        # 'solver':['lbfgs','sgd','adam'],
        'hidden_layer_sizes':hidden_layer
         }
]
begin_time=time()
grid = GridSearchCV(clf, param_dist, cv=5, n_jobs=-1)
grid.fit(x, y)
print("Best parameter",grid.best_params_)
print("Best scores:",grid.best_score_)
print("Best model:",grid.best_estimator_)
end_time=time()
print(end_time-begin_time)
