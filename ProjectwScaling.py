# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 17:02:02 2022

@author: ramra
"""

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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
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

#check_missing_df = df.isna()

#check_missing_df.to_csv("datamiss.csv")

#number_missing = df.isnull().sum()
    
#number_missing.to_csv("datamiss.csv")

result = df.isna().mean()

df_consol = df.loc[: , result < .1]

# dropping additional unneeded columns

df_consol.drop(['BidsManual', 'ListedOnUTC', 'LoanApplicationStartedDate', 'MaturityDate_Original'], axis=1, inplace = True)


df_1 = df_consol

df_1_target_popped = df_1.pop('DefaultDate')

df_1['Defaulted'] = df_1_target_popped 

df.drop(['FirstPaymentDate', 'LastPaymentOn'] , axis = 1, inplace =True)

max1 = df_1['UseOfLoan'].max() + 1
max2 = df_1['Education'].max() + 1
max3 = df_1['MaritalStatus'].max() + 1
max4 = df_1['EmploymentStatus'].max() + 1
max5 = df_1['OccupationArea'].max() + 1
max6 = df_1['HomeOwnershipType'].max() + 1

print(max1, max2, max3, max4, max5, max6)

df_1.loc[df_1['UseOfLoan'] < -0.5, 'UseOfLoan'] = 9
df_1.loc[df_1['Education'] < 0, 'Education'] = max2
df_1.loc[df_1['MaritalStatus'] < 0, 'MaritalStatus'] = max3
df_1.loc[df_1['EmploymentStatus'] < 0, 'EmploymentStatus'] = max4
df_1.loc[df_1['OccupationArea'] < 0, 'OccupationArea'] = max5
df_1.loc[df_1['HomeOwnershipType'] < 0, 'HomeOwnershipType'] = max6

df_1 = df_1.dropna()

#df_1.drop(columns = 'UseOfLoan')

df_1.to_csv("initialmodel1.csv")

target_name = "Defaulted"

df2= df_1.drop(columns=[target_name])

sc = MinMaxScaler()

# get numeric data

cols = ['AppliedAmount', 'Amount', 'Interest', 'LoanDuration', 'MonthlyPayment', 'IncomeFromPrincipalEmployer', 
          'IncomeFromPension' , 'IncomeFromFamilyAllowance' , 'IncomeFromSocialWelfare', 'IncomeFromLeavePay',
          'IncomeFromChildSupport', 'IncomeOther', 'IncomeTotal', 'ExistingLiabilities', 'LiabilitiesTotal',
          'DebtToIncome', 'FreeCash', 'MonthlyPaymentDay', 'PlannedInterestTillDate' , 'ExpectedLoss',
          'LossGivenDefault', 'ExpectedReturn', 'ProbabilityOfDefault', 'PrincipalOverdueBySchedule', 
          'PrincipalPaymentsMade', 'InterestAndPenaltyPaymentsMade', 'PrincipalBalance', 'AmountOfPreviousLoansBeforeLoan', 'Age' ]

num_d = df2[cols]

# update the cols with their normalized values
df2[num_d.columns] = sc.fit_transform(num_d)

df2['Defaulted'] = df_1_target_popped 

print(df2.head())

# now let us check for dummy encoding for categorical variables

dummies = pd.get_dummies(df2['NewCreditCustomer'], prefix = 'NewCreditCustomer', drop_first = True)

df2 = pd.concat([df2, dummies] , axis = 1)

df2.drop('NewCreditCustomer', axis = 1, inplace =True)

dummies1 = pd.get_dummies(df2['Country'], prefix = 'Country', drop_first = True)

df2 = pd.concat([df2, dummies1] , axis = 1)

df2.drop('Country', axis = 1, inplace =True)

# Unique Values in EmploymentDurationCurrentEmployer

Cur_empl_duration = list(df2['EmploymentDurationCurrentEmployer'].unique())

print(Cur_empl_duration)

dummies2 = pd.get_dummies(df2['EmploymentDurationCurrentEmployer'], prefix = 'EmploymentDurationCurrentEmployer', 
                           dummy_na = True, drop_first = True)

df2 = pd.concat([df2, dummies2] , axis = 1)

df2.drop('EmploymentDurationCurrentEmployer', axis = 1, inplace =True)


dummies3 = pd.get_dummies(df2['ActiveScheduleFirstPaymentReached'], prefix = 'ActiveScheduleFirstPaymentReached', 
                           dummy_na = True, drop_first = True
                            )

df2 = pd.concat([df2, dummies3] , axis = 1)

dummies4 = pd.get_dummies(df2['Rating'], prefix = 'Rating', 
                           dummy_na = True, drop_first = True
                            )

df2 = pd.concat([df2, dummies4] , axis = 1)

dummies5 = pd.get_dummies(df2['Status'], prefix = 'Status', 
                           dummy_na = True, drop_first = True
                            )

df2 = pd.concat([df2, dummies5] , axis = 1)

dummies6 = pd.get_dummies(df2['Restructured'], prefix = 'Restructured', 
                           dummy_na = True, drop_first = True
                            )

df2 = pd.concat([df2, dummies6] , axis = 1)

df2.drop(['ActiveScheduleFirstPaymentReached', 'Rating', 'Status', 'Restructured'], axis = 1, inplace =True)

# convert strings to datatime object datatype

# check the reason for coercion for the MaturityDate_Last Column

df2['LoanDate'] = pd.to_datetime(df2['LoanDate'], format = '%Y-%m-%d')

df2['MaturityDate_Last'] = pd.to_datetime(df2['MaturityDate_Last'],  
                                                         errors = 'coerce', format ='%Y-%m-%d')

df2['diff_days'] = (df2['MaturityDate_Last'] - df2['LoanDate']) / np.timedelta64(1, 'D')

df2.drop(['LoanDate', 'MaturityDate_Last'], axis = 1, inplace =True)

df2.drop(['FirstPaymentDate', 'LastPaymentOn'], axis = 1, inplace =True)

print(df2.head())

df2.to_csv("initialmodel2.csv")