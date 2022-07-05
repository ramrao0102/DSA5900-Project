# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:09:20 2022

@author: ramra
"""
# Next series of python files
# present the model coding in python

# RANDOM FOREST
# WITH TUNING

import pandas as pd
import numpy as np
import psycopg2
import csv
from random import seed
from csv import reader 
import random
import matplotlib.pyplot as plt
from math import exp
from math import pi
from math import sqrt
from random import random
import seaborn as sns
import csv

#sklearn Imports

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV    


filename = 'initialmodel2.csv'

df = pd.read_csv(
        filename, on_bad_lines="skip", engine="python"
    )

df1 = df.pop('Defaulted')

df['Defaulted'] = df1

df.drop(['Unnamed: 0'] , axis = 1, inplace =True)

df = df.dropna()

#df.drop(['FirstPaymentDate', 'LastPaymentOn'], axis = 1, inplace =True)
       
print(df.head())

# Split dataframe into X and y

X = df.iloc[:, :-1]

Y = df.iloc[:,-1].astype(int)

# Split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2)

print(X_train.head(), len(X_train))

print(X_test.head(), len(X_test))

print(y_train.head(), len(y_train))

print(y_test.head(), len(y_test))

# Split train into train1 and val1

X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, random_state = 1, test_size = 0.15)

print(X_train1.head(), len(X_train1))

print(X_val1.head(), len(X_val1))

print(y_train1.head(), len(y_train1))

print(y_val1.head(), len(y_val1))

# Split dataframe into X and y

X = df.iloc[:, :-1]

Y = df.iloc[:,-1].astype(int)

# Split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2)

print(X_train.head(), len(X_train))

print(X_test.head(), len(X_test))

print(y_train.head(), len(y_train))

print(y_test.head(), len(y_test))

# Split train into train1 and val1

X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, random_state = 1, test_size = 0.15)

print(X_train1.head(), len(X_train1))

print(X_val1.head(), len(X_val1))

print(y_train1.head(), len(y_train1))

print(y_val1.head(), len(y_val1))

# AdaBoost Model Fitting with GridSearchCV
# GridSearch with varying alpha values
# 5 fold cross validation is being checked

rf_Pipeline = Pipeline([('rf', RandomForestClassifier())])

param_grid = [    
    
    {'rf__n_estimators' : [50, 100, 200],
     
    'rf__criterion' : ['gini', 'entropy'],
    
    'rf__max_features' : ['sqrt', 'log2'],
    
     
    }

]

gs_rf = GridSearchCV(rf_Pipeline, param_grid, cv = 5, return_train_score = True, verbose =2)

gs_rf  = gs_rf.fit(X_train, y_train)

print(gs_rf.estimator.get_params())

print(gs_rf.best_index_)

print(gs_rf.best_params_)

cv_results = gs_rf.cv_results_

# print results of cross validation training

results_df = pd.DataFrame(
                            {'rank_cv' : cv_results['rank_test_score'],
                            'params': cv_results['params'],
                            'cv_score(mean_cv)' : cv_results['mean_test_score'],
                            'cv_score(std_cv)': cv_results['std_test_score'],
                            'cv_score(mean_train)' : cv_results['mean_train_score'],
                            'cv_score(std_train)' : cv_results['std_train_score']}
                            )

list1 =[]

for i in results_df.index:
    list1.append(str(results_df['params'][i]['rf__n_estimators']) + ',' + str(results_df['params'][i]['rf__criterion'])
                 + ',' + str(results_df['params'][i]['rf__max_features']))
    

results_df = results_df.join(pd.DataFrame({'params1': list1}))

results_df = results_df.sort_values(by = ['rank_cv'], ascending = True)

results_df.to_csv("RFResultsCV.csv")

plt.plot(results_df['cv_score(mean_train)'], results_df['params1'], label="Train")

plt.plot(results_df['cv_score(mean_cv)'], results_df['params1'], label = "CV")

plt.xlabel('Accuracy')
plt.ylabel('Model Parameter')

plt.legend(loc="upper right")


plt.show()

pd.set_option('display.max_colwidth', 100)

print(results_df)

results_df.to_csv("RandomForestResults.csv")

best_gs_rf_test_score =gs_rf.score(X_test,  y_test)

print(best_gs_rf_test_score)

y_predict2 = gs_rf.predict(X_test)
mse2 = mean_squared_error(y_predict2, y_test, squared=False)

print(mse2)

accuracy1 = accuracy_score(y_test, y_predict2)
precision1 = precision_score(y_test, y_predict2)
recall1 = recall_score(y_test, y_predict2)
F1_score = f1_score(y_test, y_predict2)
confusion_mat_test = confusion_matrix(y_test, y_predict2)

print(accuracy1, precision1, recall1, F1_score)
print(confusion_mat_test)

auc= roc_auc_score(y_test, y_predict2)

print(auc)

#Model with Best Params, this warrants a recheck.
# I picked the best model from GridSearchCV and retrained on the same set as I was not able to retrieve from GridSearchCV

rf_clf = RandomForestClassifier(criterion='entropy', n_estimators = 200, max_features = 'auto')

rf_clf.fit(X_train, y_train)

y_predict3 = rf_clf.predict(X_test)
mse3 = mean_squared_error(y_predict3, y_test, squared=False)

print(mse3)

from IPython.display import display, HTML    

feature_array = np.round(rf_clf.feature_importances_.ravel(),3)

names = X_train.columns

print(type(rf_clf.feature_importances_))

def report_coef(names, coef):
    r = pd.DataFrame( { 'coef': coef, 'more_imp': coef>=0.01  }, index = names )
    r = r.sort_values(by=['coef'])
    r.to_csv("RFFeatureImp.csv")
    display(r)
   
    data_range = r[(r['coef'] >= 0.005 )]
    ax = data_range['coef'].plot(kind='barh', color=data_range['more_imp'].map(
        {True: 'r', False: 'b'}), figsize=(11, 8))
    
    for container in ax.containers:
        ax.bar_label(container)

    
report_coef(
  names,
  feature_array) 
