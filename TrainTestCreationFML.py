# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 12:42:39 2022

@author: ramra
"""

# Train, Test Creation for Federated ML

import pandas as pd
import numpy as np

import csv

from random import seed
from csv import reader 

filename = 'initialmodel2.csv'

df = pd.read_csv(
        filename, on_bad_lines="skip", engine="python"
    )

df1 = df.pop('Defaulted')

df['Defaulted'] = df1

df.drop(['Unnamed: 0'] , axis = 1, inplace =True)

df = df.dropna()

print(df.head())


df_sub = df.sample(frac = 0.05, random_state=2)

print(len(df_sub))

count = 0

for i in df_sub.index:
    if df_sub['Defaulted'][i] == 0:
        count += 1 
        
print(count/len(df_sub))


df_train = df_sub.sample(frac = 0.80, random_state=2)

df_test = pd.concat([df_sub, df_train])

df_test = df_test.drop_duplicates(keep=False)

count = 0

for i in df_train .index:
    if df_train['Defaulted'][i] == 0:
        count += 1 
        
print(count/len(df_train))

count = 0

for i in df_test .index:
    if df_test['Defaulted'][i] == 0:
        count += 1 
        
print(count/len(df_test))

df3 = df_train.merge(df_test, how = 'inner' ,indicator=False)

print(df3)

df_train.to_csv("Train.csv")

df_test.to_csv("Test.csv")




