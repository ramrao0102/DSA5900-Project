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
from sklearn.decomposition import PCA
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

df_1.loc[df_1['UseOfLoan'] < -0.5, 'UseOfLoan'] = 9 #not sure what is happening here yet!
df_1.loc[df_1['Education'] < 0, 'Education'] = max2
df_1.loc[df_1['MaritalStatus'] < 0, 'MaritalStatus'] = max3
df_1.loc[df_1['EmploymentStatus'] < 0, 'EmploymentStatus'] = max4
df_1.loc[df_1['OccupationArea'] < 0, 'OccupationArea'] = max5
df_1.loc[df_1['HomeOwnershipType'] < 0, 'HomeOwnershipType'] = max6

df_1 = df_1.dropna()

#df_1.drop(columns = 'UseOfLoan')

#df_1.to_csv("initialmodel1.csv")

target_name = "Defaulted"

#df2= df_1

df2 = df_1.reset_index(drop = True)    # Apply reset_index function

sc = StandardScaler()

# get numeric data

cols = ['AppliedAmount', 'Amount', 'Interest', 'LoanDuration', 'MonthlyPayment', 'IncomeFromPrincipalEmployer', 
          'IncomeFromPension' , 'IncomeFromFamilyAllowance' , 'IncomeFromSocialWelfare', 'IncomeFromLeavePay',
          'IncomeFromChildSupport', 'IncomeOther', 'IncomeTotal', 'ExistingLiabilities', 'LiabilitiesTotal',
          'DebtToIncome', 'FreeCash', 'MonthlyPaymentDay', 'PlannedInterestTillDate' , 'ExpectedLoss',
          'LossGivenDefault', 'ExpectedReturn', 'ProbabilityOfDefault', 'PrincipalOverdueBySchedule', 
          'PrincipalPaymentsMade', 'InterestAndPenaltyPaymentsMade', 'PrincipalBalance', 'AmountOfPreviousLoansBeforeLoan', 'Age', target_name ]


df2 = df2.iloc[0:5000]

print(df2)

cols1 = ['AppliedAmount', 'Amount', 'Interest', 'LoanDuration', 'MonthlyPayment', 'IncomeFromPrincipalEmployer', 
          'IncomeFromPension' , 'IncomeFromFamilyAllowance' , 'IncomeFromSocialWelfare', 'IncomeFromLeavePay',
          'IncomeFromChildSupport', 'IncomeOther', 'IncomeTotal', 'ExistingLiabilities', 'LiabilitiesTotal',
          'DebtToIncome', 'FreeCash', 'MonthlyPaymentDay', 'PlannedInterestTillDate' , 'ExpectedLoss',
          'LossGivenDefault', 'ExpectedReturn', 'ProbabilityOfDefault', 'PrincipalOverdueBySchedule', 
          'PrincipalPaymentsMade', 'InterestAndPenaltyPaymentsMade', 'PrincipalBalance', 'AmountOfPreviousLoansBeforeLoan', 'Age']


num_d = df2[cols1]

print(type(num_d))

# update the cols with their normalized values
num_d[num_d.columns] = sc.fit_transform(num_d)

#df2['Defaulted'] = df_1_target_popped 

print(num_d)

pca = PCA(n_components=5)
principalComponents = pca.fit_transform(num_d)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3',
                          'principal component 4', 'principal component 5'])

print(principalDf)

finalDf = pd.concat([principalDf, df2[target_name]], axis = 1)

print(finalDf)

print(pca.explained_variance_ratio_)

x_axis = [1, 2, 3, 4, 5 ]

plt.figure(figsize=(4, 4))
plt.plot(x_axis ,  pca.explained_variance_ratio_)

plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()


plt.figure(figsize=(11,11))

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = .6/(xs.max() - xs.min())
    scaley =.8/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

myplot(principalComponents[:,0:2],np.transpose(pca.components_[0:2, :]),list(num_d.columns))
plt.show()