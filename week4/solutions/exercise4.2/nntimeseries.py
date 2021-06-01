import numpy as np 

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets 
from sklearn.metrics import mean_squared_error  

from sklearn.model_selection import train_test_split 
from sklearn import metrics 

#keras 

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers

from keras.layers import Dense
from keras.models import Sequential

from tensorflow.keras.callbacks import Callback

import random


from sklearn.metrics import roc_curve, auc

 

def read_data(run_num):
    #Source - raw and processed data :  https://github.com/sydney-machine-learning/Bayesianneuralnet_stockmarket/tree/master/code/dataset
    # five inputs (window size of 5) for 5 steps ahead (MMM dataset) https://finance.yahoo.com/quote/MMM/
    #code to process raw data: https://github.com/sydney-machine-learning/Bayesianneuralnet_stockmarket/blob/master/code/data.py
    data_in = genfromtxt("https://raw.githubusercontent.com/sydney-machine-learning/Bayesianneuralnet_stockmarket/master/code/dataset/MMM8_train.txt", delimiter=" ")
    data_inputx = data_in[:,0:5] # all features 0, 1, 2, 3, 4, 5, 6, 7 

    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
    #transformer = Normalizer().fit(data_inputx)  # fit does nothing.
    #data_inputx = transformer.transform(data_inputx)
    data_inputy = data_in[:,5:10] # this is target - so that last col is selected from data

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=run_num)

    return x_train, x_test, y_train, y_test

 
    
def nn(x_train, x_test, y_train, y_test, type_model, hidden):
 

    timesteps = 5 # window size
    steps_ahead = 5 

    if type_model ==0: #keras Adam
        nn = keras.Sequential()
        nn.add(layers.Dense(5, input_dim=timesteps, activation='relu'))
        nn.add(layers.Dense(steps_ahead, activation='sigmoid'))
        nn.compile(loss=keras.losses.binary_crossentropy,optimizer='adam', metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.mae,keras.metrics.mape])
         
    else:
        print('no model')    
  
    history = nn.fit(x_train, y_train, epochs=500, batch_size=32, verbose=0)#callbacks=[fa_test_his])
    y_pred_test = nn.predict(x_test)
    n = len(y_test)
    MAE = sum(np.abs(y_test - y_pred_test)) / n 
    RMSE = np.sqrt(sum(np.square(y_test - y_pred_test)) / n) 
    MAPE=sum(np.abs((y_test - y_pred_test) / (y_test + 1e-6))) / n * 100  
   
    return RMSE, y_test, y_pred_test


def main(): 

    max_expruns = 2

    Adam_all = np.zeros(max_expruns) 
      
    max_hidden = 10
  
    for hidden in range(6,max_hidden, 2):
 
        for run_num in range(0,max_expruns): 
    
            x_train, x_test, y_train, y_test = read_data(0)   

            acc_adam, y_test, y_pred_test = nn(x_train, x_test, y_train, y_test, 0, hidden) #Adam 
             
            Adam_all[run_num] = acc_adam
   
        print(Adam_all, hidden,' RMSE all runs')
        print(np.mean(Adam_all), hidden, ' mean ')
        print(np.std(Adam_all), hidden, ' std')
 
 
 
    #next try a paragraph to describe your results and discuss which models are better to use.
    #repeat for another dataset
    # you can save results to a file as well


 


if __name__ == '__main__':
     main() 
