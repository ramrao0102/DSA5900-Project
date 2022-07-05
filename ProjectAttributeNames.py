# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 18:36:00 2022

@author: ramra
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 17:58:28 2022

@author: ramra
"""

# CODE TO PRINT ATTRIBUTE NAMES

# This is Ramkishore Rao's DSA 5900 practicuum project

import pandas as pd
import csv
from csv import reader 
import csv

# Reading Loan Dataset File

filename = 'LoanData.csv'

df = pd.read_csv(
        filename, on_bad_lines="skip", engine="python"
    )

df.rename(columns = {'DefaultDate':'Defaulted'}, inplace = True)

df1 = df.pop('Defaulted')

df['Target Class: Defaulted'] = df1

count = 1

print("Feature No", "Feature Name", sep = '\t')

print("")

for i in df.columns:
    print(count, i, sep='\t\t\t')
    count += 1