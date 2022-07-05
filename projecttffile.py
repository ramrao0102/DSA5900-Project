# -*- coding: utf-8 -*-
"""ProjectTFFile.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LPP9p-DQC9Q9zQBOk_3ED3gLEjFmx0od

This is Ramkishore Rao's Project - Application of Tensor Flow and Keras for Loan Dataset
"""

# TENSORFLOW/KERAS
# DEFAULT

import tensorflow.keras
from tensorflow . keras .models import Sequential
from tensorflow . keras . layers import Dense , Activation
from tensorflow . keras . callbacks import EarlyStopping
from sklearn . model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn import metrics
import sklearn
from sklearn.model_selection import train_test_split
import io
import requests
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt

df = pd.read_csv('/content/initialmodel2.csv', on_bad_lines="skip", engine="python")

df.head()
len(df)

"""# New Section"""

df1 = df.pop('Defaulted')

df['Defaulted'] = df1

df.drop(['Unnamed: 0'] , axis = 1, inplace =True)

df = df.dropna()
print(df.head())

"""Split dataframe into X and y"""

X = df.iloc[:, :-1]

Y = df.iloc[:,-1].astype(int)

# Split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2)

"""Split train into train1 and val1"""

X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, random_state = 1, test_size = 0.15)

model = Sequential()
model.add(Dense(100, input_dim=X_train1.shape[1], activation='relu',
                kernel_initializer='random_normal'))
model.add(Dense(50,activation='relu',kernel_initializer='random_normal'))
model.add(Dense(25,activation='relu',kernel_initializer='random_normal'))
model.add(Dense(1,activation='sigmoid',kernel_initializer='random_normal'))
model.compile(loss='binary_crossentropy', 
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics =['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
    patience=5, verbose=1, mode='auto', restore_best_weights=True)

model.fit(X_train1,y_train1,validation_data=(X_test,y_test),
          callbacks=[monitor],verbose=2,epochs=1000)

pred = model.predict(X_test)
pred

mse2 = mean_squared_error(pred, y_test, squared=False)

mse2

pred1 = np.round(pred) # this takes continues output and transforms to binary values of 0 and 1

pred1 # this is the output target value array for the test dataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

accuracy1 = accuracy_score(y_test, pred1)
precision1 = precision_score(y_test, pred1)
recall1 = recall_score(y_test, pred1)
F1_score = f1_score(y_test, pred1)
confusion_mat_test = confusion_matrix(y_test, pred1)

confusion_mat_test

accuracy1

precision1

recall1

F1_score

auc= roc_auc_score(y_test, pred)

print(auc)

# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

pred = model.predict(X_test)
fper, tper, thresholds = roc_curve(y_test, pred) 
plot_roc_curve(fper, tper)

