# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:09:20 2022

@author: ramra
"""
# Next series of python files
# present the model coding in python

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

# MNB Model Fitting with GridSearchCV
# GridSearch with varying alpha values
# 5 fold cross validation is being checked

mnb_Pipeline = Pipeline([('mnb', MultinomialNB())])

param_grid = {'mnb__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1]}

gs_mnb = GridSearchCV(mnb_Pipeline, param_grid, cv = 5, return_train_score = True, verbose =2)

gs_mnb = gs_mnb.fit(X_train, y_train)

print(gs_mnb.estimator.get_params())

print(gs_mnb.best_index_)

print(gs_mnb.best_params_)

cv_results = gs_mnb.cv_results_

# print results of cross validation training

results_df = pd.DataFrame(
                            {
                             'rank_cv' : cv_results['rank_test_score'],
                             'params': cv_results['params'],
                             'cv_score(mean_cv)' : cv_results['mean_test_score'],
                             'cv_score(std_cv)': cv_results['std_test_score'],
                             'cv_score(mean_train)' : cv_results['mean_train_score'],
                             'cv_score(std_train)' : cv_results['std_train_score']
                             }
                            )

pd.set_option('display.max_colwidth', 100)

list1 = []

for i in results_df.index:
    list1.append(str(results_df['params'][i]['mnb__alpha']))

results_df = results_df.join(pd.DataFrame({'params1': list1}))

results_df = results_df.sort_values(by = ['rank_cv'], ascending = True)

results_df.to_csv("MNBResultsCV.csv")

plt.plot(results_df['cv_score(mean_train)'], results_df['params1'], label="Train")

plt.plot(results_df['cv_score(mean_cv)'], results_df['params1'], label = "CV")

plt.xlabel('Accuracy')

plt.ylabel('Model Parameter')

plt.legend(loc="upper right")

plt.xlim(0.8372,0.8378)


plt.show()

print(results_df)

best_gs_mnb_test_score = gs_mnb.score(X_test,  y_test)

print(best_gs_mnb_test_score)

y_predict2 = gs_mnb.predict(X_test)
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

# function for ROC Curve Plotting

def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
probs = gs_mnb.predict_proba(X_test)  
probs = probs[:, 1]  
fper, tper, thresholds = roc_curve(y_test, probs) 
plot_roc_curve(fper, tper)
 
#Model with Best Params, this warrants a recheck.
# I picked the best model from GridSearchCV and retrained on the same set as I was not able to retrieve from GridSearchCV

mnb_clf = MultinomialNB(alpha= 1.0)

mnb_clf.fit(X_train, y_train)

y_predict3 = mnb_clf.predict(X_test)
mse3 = mean_squared_error(y_predict3, y_test, squared=False)

print(mse3)

from IPython.display import display, HTML    

coeff_array = mnb_clf.feature_log_prob_
                       
names = X_train.columns       

print(coeff_array)   

list1 =[]

for i in range(len(coeff_array[0])):
    list1.append(abs(coeff_array[0][i] - coeff_array[1][i]))
    if coeff_array[0][i] >=coeff_array[1][i]:
        print(names[i], coeff_array[0][i], "Class_0_dominates")
              
    else:
        print(names[i], coeff_array[1][i], "Class_1_dominates")
        

def report_coef(names, coef):
    r = pd.DataFrame( { 'coef': coef, 'more_imp': coef>=0.01, 'names': names  }, index = names )

    r = r.sort_values(by=['coef'])
    r.to_csv("MNBImp.csv")
    display(r)
   
    data_range = r[(r['coef'] >= 0.5 )]
    
    data_range = data_range[data_range['names'].str.contains("nan") == False]
    
    data_range.drop(['names'], axis = 1)
    
    ax = data_range['coef'].plot(kind='barh', color=data_range['more_imp'].map(
        {True: 'r', False: 'b'}), figsize=(11, 8))
    
    for container in ax.containers:
        ax.bar_label(container)

array1 = np.asarray(list1)

array1 = np.round(array1, 2)
        
report_coef(
  names,
  array1)
                 
              




