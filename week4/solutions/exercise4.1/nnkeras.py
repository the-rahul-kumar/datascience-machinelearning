import numpy as np 

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

#keras 

from keras.layers import Dense
from keras.models import Sequential

from keras.regularizers import l2

import random


from sklearn.metrics import roc_curve, auc

def read_data(run_num):
    #Source:  Pima-Indian diabetes dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv
    data_in = genfromtxt("pima-indians-diabetes.csv", delimiter=",")
    data_inputx = data_in[:,0:8] # all features 0, 1, 2, 3, 4, 5, 6, 7 

    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
    #transformer = Normalizer().fit(data_inputx)  # fit does nothing.
    #data_inputx = transformer.transform(data_inputx)
    data_inputy = data_in[:,-1] # this is target - so that last col is selected from data

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=run_num)

    return x_train, x_test, y_train, y_test

 
    
def keras_nn(x_train, x_test, y_train, y_test, type_model, hidden, learn_rate, run_num):
 
    #https://keras.io/api/models/model_training_apis/

    #note that keras model on own ensures that every run begins with different initial 
    #weights so run_num is not needed 

    if type_model ==0: #SGD
        #nn = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=run_num, max_iter=100,solver='sgd',  learning_rate_init=learn_rate )
        model = Sequential()
        model.add(Dense(hidden, input_dim=x_train.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='sgd',  metrics=['accuracy'])
    
    elif type_model ==1: #Adam
        #nn = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=run_num, max_iter=100,solver='adam', learning_rate_init=learn_rate)
        model = Sequential()
        model.add(Dense(hidden, input_dim=x_train.shape[1], activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    elif type_model ==2: #SGD with 2 hidden layers
        #nn = MLPClassifier(hidden_layer_sizes=(hidden,hidden), random_state=run_num, max_iter=100,solver='sgd',learning_rate='constant', learning_rate_init=learn_rate)
        #hidden_layer_sizes=(hidden,hidden, hidden) would implement 3 hidden layers
        model = Sequential()
        model.add(Dense(hidden, input_dim=x_train.shape[1], activation='sigmoid')) 
        model.add(Dense(hidden, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    else:
        print('no model')    


    
    # Fit model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=500, batch_size=10, verbose=0)

    # Evaluate the model
    #https://keras.io/api/models/model_training_apis/
    _, acc_train = model.evaluate(x_train, y_train, verbose=0)
    _, acc_test = model.evaluate(x_test, y_test, verbose=0)
    #print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    # Plot history
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.savefig(str(type_model)+'nodp.png') 
    plt.clf()
   
    #auc = roc_auc_score(y_pred, y_test, average=None) 
    return acc_test #,acc_train


def main(): 

    max_expruns = 2

    SGD_all = np.zeros(max_expruns) 
    Adam_all = np.zeros(max_expruns) 
    
    SGD2_all = np.zeros(max_expruns)  
    max_hidden = 8 # 

    learn_rate = 0.01 


    #for learn_rate in range(0.1,1, 0.2):
    
    for hidden in range(6,max_hidden, 2): # only cover 6 hidden neurons for now
 
        for run_num in range(0,max_expruns): 
    
            x_train, x_test, y_train, y_test = read_data(0)   
            
            acc_sgd = keras_nn(x_train, x_test, y_train, y_test, 0, hidden, learn_rate, run_num) #SGD
            acc_adam = keras_nn(x_train, x_test, y_train, y_test, 1, hidden, learn_rate, run_num) #Adam 
            acc_sgd2 = keras_nn(x_train, x_test, y_train, y_test, 2, hidden, learn_rate,  run_num) #SGD2
           
            SGD_all[run_num] = acc_sgd
            Adam_all[run_num] = acc_adam

            SGD2_all[run_num] = acc_sgd2 # two hidden layers
        
        print(SGD_all, hidden,' SGD_all')
        print(np.mean(SGD_all), hidden, ' mean SGD_all')
        print(np.std(SGD_all), hidden, ' std SGD_all')

        print(Adam_all, hidden,' Adam_all')
        print(np.mean(Adam_all), hidden, ' Adam _all')
        print(np.std(Adam_all), hidden, ' Adam _all')

        print(SGD2_all, hidden,' SGD2_all')

        # you can also print  for Adam

 
 
    #next try a paragraph to describe your results and discuss which models are better to use.
    #repeat for another dataset
    # you can save results to a file as well


 


if __name__ == '__main__':
     main() 
