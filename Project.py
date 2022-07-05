
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 17:58:28 2022

@author: ramra
"""

# PROCESSES LOAN DATASET
# CREATES CORRELATION MATRIX
# CREATES AN INITIAL FILE FOR LOADING
# BUT IT REQUIRED PROCESSING, SEE PROJECT1.PY

# This is Ramkishore Rao's DSA 5900 practicuum project

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

seed(500)

# Reading Loan Dataset File

filename = 'LoanData.csv'

df = pd.read_csv(
        filename, on_bad_lines="skip", engine="python"
    )

# Print First 5 Rows of Dataframe

print(df.head())

# Now cleaning the dataframe
# Remove Unnecessary Columns

df.drop(columns= ['ReportAsOfEOD', 'LoanId', 'LoanNumber',
                  'BiddingStartedOn', 'BidsPortfolioManager', 'BidsApi',
                  'PartyId', 'ApplicationSignedHour', 'ApplicationSignedWeekday',
                  'County', 'City', 'EmploymentPosition', 'EL_V0', 'Rating_V0'],  inplace=True) 

print(df.head())

df[['DefaultDate']] = df[['DefaultDate']].fillna(value=0)

df.loc[df['DefaultDate'] != 0, 'DefaultDate'] = 1

check_missing_df = df.isna()

# checks the dataframe to see of there are missing values or no

check_missing_df.to_csv("datamiss.csv")

number_missing = df.isnull().sum()

# this tells us number missing in each column
    
number_missing.to_csv("datamiss1.csv")

result = df.isna().mean()

result.to_csv("missingresult.csv")

print(result)

df_consol = df.loc[: , result < .1]

# dropping additional unneeded columns

df_consol.drop(['BidsManual', 'ListedOnUTC', 'LoanApplicationStartedDate', 'MaturityDate_Original'], axis=1, inplace = True  )

# now let us check for dummy encoding for categorical variables

dummies = pd.get_dummies(df_consol['NewCreditCustomer'], prefix = 'NewCreditCustomer', drop_first = True)

df_consol = pd.concat([df_consol, dummies] , axis = 1)

df_consol.drop('NewCreditCustomer', axis = 1, inplace =True)

dummies1 = pd.get_dummies(df_consol['Country'], prefix = 'Country', drop_first = True)

df_consol = pd.concat([df_consol, dummies1] , axis = 1)

df_consol.drop('Country', axis = 1, inplace =True)

# Unique Values in EmploymentDurationCurrentEmployer

Cur_empl_duration = list(df['EmploymentDurationCurrentEmployer'].unique())

print(Cur_empl_duration)

dummies2 = pd.get_dummies(df_consol['EmploymentDurationCurrentEmployer'], prefix = 'EmploymentDurationCurrentEmployer', 
                           dummy_na = True, drop_first = True)

df_consol = pd.concat([df_consol, dummies2] , axis = 1)

df_consol.drop('EmploymentDurationCurrentEmployer', axis = 1, inplace =True)


dummies3 = pd.get_dummies(df_consol['ActiveScheduleFirstPaymentReached'], prefix = 'ActiveScheduleFirstPaymentReached', 
                           dummy_na = True, drop_first = True
                            )

df_consol = pd.concat([df_consol, dummies3] , axis = 1)

dummies4 = pd.get_dummies(df_consol['Rating'], prefix = 'Rating', 
                           dummy_na = True, drop_first = True
                            )

df_consol = pd.concat([df_consol, dummies4] , axis = 1)

dummies5 = pd.get_dummies(df_consol['Status'], prefix = 'Status', 
                           dummy_na = True, drop_first = True
                            )

df_consol = pd.concat([df_consol, dummies5] , axis = 1)

dummies6 = pd.get_dummies(df_consol['Restructured'], prefix = 'Restructured', 
                           dummy_na = True, drop_first = True
                            )

df_consol = pd.concat([df_consol, dummies6] , axis = 1)

df_consol.drop(['ActiveScheduleFirstPaymentReached', 'Rating', 'Status', 'Restructured'], axis = 1, inplace =True)

# convert strings to datatime object datatype

# check the reason for coercion for the MaturityDate_Last Column

df_consol['LoanDate'] = pd.to_datetime(df_consol['LoanDate'], format = '%Y-%m-%d')

df_consol['MaturityDate_Last'] = pd.to_datetime(df_consol['MaturityDate_Last'],  
                                                         errors = 'coerce', format ='%Y-%m-%d')

df_consol['diff_days'] = (df_consol['MaturityDate_Last'] - df_consol['LoanDate']) / np.timedelta64(1, 'D')

df_consol.drop(['LoanDate', 'MaturityDate_Last'], axis = 1, inplace =True)

print(df_consol.dtypes)

print(df_consol.head(10))

df_consol.to_csv("dataconsol.csv")


# print(df_consol.head())

# Number Missing in consolidated dataframe

number_missing = df_consol.isnull().sum()

missing_df = pd.DataFrame(number_missing)

missing_df.columns = ['Missing_Number']

#missing_df = pd.DataFrame(missing_df, columns = column_name) 

number_missing.to_csv("datamiss2.csv")

#print(missing_df.head(60))  

# Missing Values Bar Chart
# plot only if missing

only_miss_df = missing_df[missing_df['Missing_Number'] != 0]

only_miss_df.to_csv("onlymiss.csv")

ax = only_miss_df.plot.barh(figsize=(12, 8))

ax.bar_label(ax.containers[0])

# Now next steps are to check multi collinearity and correlation matrices
# Question is how to check if column values are real and not categorical without looking at the data?
# Not sure

df_for_correl = df_consol

df_for_correl.drop(['VerificationType','ActiveScheduleFirstPaymentReached_nan', 
                    'Rating_nan', 'Status_nan', 'Restructured_nan',
                    'LanguageCode', 'Age', 'Gender', 'IncomeTotal', 'EmploymentDurationCurrentEmployer_nan'], axis = 1, inplace =True)

df_for_correl.corr().to_csv("corr_matrix.csv")

print(df_for_correl.corr())

corr_dict =df_for_correl.corr().to_dict('dict')

#print(corr_dict)


def iterate_nest_Dict(data_dict):
     
    
    for key, value in data_dict.items():
         
        if isinstance(value, dict):
            
            for key_value in  iterate_nest_Dict(value):
                yield (key, *key_value)
        else:
           
            yield (key, value)
 
# now let us attempt to iterate through the correlation matrix dictionary
# only prints correlation coefficients that exceed 0.75.

list1 =[]    
 
for key_value in iterate_nest_Dict(corr_dict):
    if key_value[0] != key_value[1]:
        if key_value[2] > 0.75:
            list1.append([key_value[0], key_value[1], key_value[2]])
            
#print(list1)     

print(len(list1))

print("Variable 1", "," , "Variable 2", "," , "Corr_Coefficient")
print("______________________________________________")

for i in range(len(list1)):
    print(list1[i][0], "," , list1[i][1], ",", round(list1[i][2],3))
    

for i in range(len(list1)):
    list1[i][2] = str(round(list1[i][2],3))

print(list1)

rows = list1
    
# prints high correlated values to csv file

filename = 'corr_file.csv'

fields = ['Variable_1', 'Variable_2', 'Value']
    
with open(filename, 'w') as csvfile:   
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)


# Next step is to fill missing values in the consolidated dataframe columns

# Columns in DataFrame with Missing Values are !
# they are the rows of only_miss_df

initial_model_df = df_consol.dropna()

count_target0 = 0

for i in initial_model_df.index:
    if (initial_model_df['DefaultDate'][i] == 0):
        count_target0 += 1
        
print(count_target0)

print(len(initial_model_df))

initial_model_df.to_csv("initialmodel.csv")

