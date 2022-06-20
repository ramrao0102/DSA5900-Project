# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 19:18:47 2022

@author: ramra
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

filename = 'initialmodel.csv'

df = pd.read_csv(
        filename, on_bad_lines="skip", engine="python"
    )

df1 = df.pop('DefaultDate')

df['Defaulted'] = df1

df.drop(['Unnamed: 0', 'FirstPaymentDate', 'LastPaymentOn'] , axis = 1, inplace =True)

print(df.head())

a = df[df.columns[0:]].corr()['Defaulted'][:]

a.to_csv("correlation.csv")