# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:09:20 2022

@author: ramra
"""
# Next series of python files
# present the model coding in python

# KS MEANS

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
from matplotlib.ticker import FormatStrFormatter

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
from sklearn.cluster import KMeans

filename = 'initialmodel2-1.csv'

df = pd.read_csv(
        filename, on_bad_lines="skip", engine="python"
    )

df1 = df.pop('Defaulted')

df['Defaulted'] = df1

df.drop(['Unnamed: 0'] , axis = 1, inplace =True)

df = df.dropna()

new_df = df

#df.drop(['FirstPaymentDate', 'LastPaymentOn'], axis = 1, inplace =True)
       
print(df.head())

# Split dataframe into X and y

X = df.iloc[:, :-1]

Y = df.iloc[:,-1].astype(int)

WCSS = []
K = range(1,13)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    WCSS.append(kmeanModel.inertia_)
    
plt.figure(figsize=(8,8))

plt.rc('font', size = 20 )

plt.plot(K, WCSS, 'bx-')
plt.xlim(0,15)
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.title('The Elbow Method showing the optimal clusters')
plt.show()

kmeans_model = KMeans(n_clusters=6, random_state=42)

kmeans_predict = kmeans_model.fit_predict(X)

centroids = kmeans_model.cluster_centers_

print(type(centroids))

vals  = []

for i in range(len(centroids[0])):
   max1 = 0
   min1 = 10000
   for j in range(centroids.shape[0]):
       if centroids[j][i] >max1:
           max1 = centroids[j][i]
       if centroids[j][i] <min1:
           min1 = centroids[j][i]
           
   vals.append((max1, min1))
   

spread = []

for i in range(len(vals)):
    spread.append(vals[i][0] -vals[i][1])

    
print(spread)

r = pd.DataFrame( { 'spread': spread, 'names': X.columns})
    
r = r.sort_values(by=['spread'], ascending = False)
r.to_csv("spread.csv")
      
print(type(kmeans_predict))

print(kmeans_predict)

new_df['pred'] = kmeans_predict

print(new_df.head())

df1 = new_df[new_df["pred"] == 0]
df2 = new_df[new_df["pred"] == 1]
df3 = new_df[new_df["pred"] == 2]
df4 = new_df[new_df["pred"] == 3]
df5 = new_df[new_df["pred"] == 4]
df6 = new_df[new_df["pred"] == 5]

plt.rc('font', size = 20 )

plt.scatter(df6.PrincipalOverdueBySchedule, df6.ProbabilityOfDefault, c = 'gray',  label = '1', alpha = 0.5)
plt.scatter(df1.PrincipalOverdueBySchedule, df1.ProbabilityOfDefault, c = 'blue', label = '2', alpha = 0.5)
plt.scatter(df2.PrincipalOverdueBySchedule, df2.ProbabilityOfDefault, c = 'green', label = '3',  alpha = 0.5)
plt.scatter(df3.PrincipalOverdueBySchedule, df3.ProbabilityOfDefault, c = 'orange', label = '4', alpha = 0.5)
plt.scatter(df4.PrincipalOverdueBySchedule, df4.ProbabilityOfDefault, c = 'red', label = '5', alpha = 0.5)
plt.scatter(df5.PrincipalOverdueBySchedule, df5.ProbabilityOfDefault, c = 'yellow', label = '6', alpha = 0.5)
plt.xlabel("PrincipalOverdueBySchedule")
plt.ylabel('ProbabilityOfDefault')
plt.title('K Means Visualization')
plt.legend()

plt.show()