# -*- coding: utf-8 -*-
"""ProjectTFFile_With_TuningLatest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PuQFSnb_P3iAXjYFPsyvQlo2Fg5BFrif

This is Ramkishore Rao's Project - Application of Tensor Flow and Keras for Loan Dataset.  This one includes sckit learn's gridsearchCV for
hyperparameter tuning
"""

# TENSORFLOW/KERAS 
# TUNED FOR HYPERPARAMETERS, 10 PCT OF DATASET

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
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

!pip install theano

df = pd.read_csv('/content/initialmodel2.csv', on_bad_lines="skip", engine="python")

df.head()
len(df)

df1 = df.pop('Defaulted')

df['Defaulted'] = df1

df.drop(['Unnamed: 0'] , axis = 1, inplace =True)

df = df.dropna()
print(df.head())

"""Split dataframe into X and y"""

df = df.iloc[0:20000] # let's take a subset of the dataset as the neural net takes long to execute

X = df.iloc[:, :-1]
X1= df.iloc[:, :-1]

Y = df.iloc[:,-1].astype(int)

# Split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2)

"""Split train into train1 and val1"""

X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, random_state = 1, test_size = 0.15)

X = X_train.to_numpy()
y = y_train.to_numpy()

len(X)

len(y)

def create_model (optimizer = 'rmsprop', init = 'glorot_uniform'):
  model = Sequential()
  model.add(Dense(100, input_dim=X.shape[1], activation='relu',
                  kernel_initializer=init))
  model.add(Dense(50,activation='relu',kernel_initializer= init))
  model.add(Dense(25,activation='relu',kernel_initializer=init))
  model.add(Dense(1,activation='sigmoid',kernel_initializer=init))

# Compile Model 

  model.compile(loss='binary_crossentropy', 
                optimizer= optimizer,
                metrics =['accuracy'])
  return model

# create model

model = KerasClassifier(build_fn = create_model, verbose = 0)

# grid search, epochs, batch size and optimizer with sckitlearn's gridsearchCV

optimizers = ['rmsprop', 'adam']
inits = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 150]
batches = [5, 20]

param_grid = dict(optimizer = optimizers, epochs = epochs, batch_size = batches, init = inits)
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv =3)
grid_result = grid.fit(X,y)

# summarize results

print("Best %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip (means, stds, params):
  print("%f (%f) with %r" % (mean, stdev, param))

# run model with best parameters from sckit learn's gridsearchCV

best_model = Sequential()
best_model.add(Dense(100, input_dim=X.shape[1], activation='relu',
                  kernel_initializer='glorot_uniform'))
best_model.add(Dense(50,activation='relu',kernel_initializer= 'glorot_uniform'))
best_model.add(Dense(25,activation='relu',kernel_initializer='glorot_uniform'))
best_model.add(Dense(1,activation='sigmoid',kernel_initializer='glorot_uniform'))

best_model.compile(loss='binary_crossentropy', 
                optimizer= tensorflow.keras.optimizers.Adam(),
                metrics =['accuracy'])

best_model.fit(X ,  y, epochs = 150, batch_size = 5)

"""Predictions from Best Model Provided Below"""

pred = best_model.predict(X_test)
pred

mse2 = mean_squared_error(pred, y_test, squared=False)

mse2

pred1 = np.round(pred) # this takes continues output and transforms to binary values of 0 and 1
pred1.shape

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

def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
probs = best_model.predict(X_test)
fper, tper, thresholds = roc_curve(y_test, probs) 
plot_roc_curve(fper, tper)

# Save Neural Network to JSON File

from keras.models import model_from_json

# Serialize model to JSON

best_tuned_model_json = best_model.to_json()

with open("model.json" , "w") as json_file:
  json_file.write(best_tuned_model_json)

# Serialize weights to HDF5

best_model.save_weights("best_model.h5")
print("Saved Model to Disk")

# Load json and create model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)


# Load weights into new Model


loaded_model.load_weights("best_model.h5")
print("Loaded model from disk")

# Evaluate Loaded Model on Test Data

loaded_model.compile(loss='binary_crossentropy', 
                optimizer= tensorflow.keras.optimizers.Adam(),
                metrics =['accuracy'])

score = loaded_model.evaluate(X_test, y_test, verbose = 0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

!pip install matplotlib --upgrade

print(loaded_model.layers[0].weights[0].shape)

list1 = []

for x in loaded_model.layers[0].weights[0]:
  a = (np.sum(abs(x)))
  a = np.round(a,2)
  list1.append(a)

array1 = np.array(list1)

X1 = X1.columns
X1 = X1.tolist()
list2 =[]

for i in range(len(X1)):
  list2.append((list1[i], X1[i]))

print (list2)



def report_coef(names, coef):
    r = pd.DataFrame( { 'coef': coef, 'more_imp': coef>=30  }, index = names )
    r = r.sort_values(by=['coef'])
    r.to_csv("BestModelNeuralNet.csv")
    display(r)
   
    data_range = r[(r['coef'] >=30 )]
    ax = data_range['coef'].plot(kind='barh', color=data_range['more_imp'].map(
        {True: 'r', False: 'b'}), figsize=(11, 8))
    
    for container in ax.containers:
        ax.bar_label(container)
    
    plt.xlabel("Sum of Absolute Values of Weights")

    
report_coef(
  X1,
  array1)

# Let's look at history of training errors in the best_model retraining

!pip install plot_keras_history
from plot_keras_history import show_history, plot_history
print(best_model.history)

print(best_model.history.history.keys())

plt.plot(best_model.history.history['accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.legend(['train'], loc = 'lower right')
plt.title("Model Accuracy")

plt.plot(best_model.history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train'], loc = 'upper right')
plt.title("Model Loss")

