import numpy as np 

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import Normalizer

from sklearn import metrics

from sklearn.metrics import roc_auc_score

import random


from sklearn.metrics import roc_curve, auc

def read_data():
    #Source:  Pima-Indian diabetes dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv

    #data_in = genfromtxt("irisprocessed.csv", delimiter=",")

    data_in = genfromtxt("pima-indians-diabetes.csv", delimiter=",")



    data_inputx = data_in[:,0:8] # all features 0, 1, 2, 3, 4, 5, 6, 7

    #data_inputx = data_in[:,[1]]  # one feature

    #data_inputx = data_in[:,[1,2]]  # two features


    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
    #transformer = Normalizer().fit(data_inputx)  # fit does nothing.
    #data_inputx = transformer.transform(data_inputx)

    data_inputy = data_in[:,-1] # this is target - so that last col is selected from data

    percent_test = 0.4

    testsize = int(percent_test * data_inputx.shape[0])
    print(testsize, ' is testsize')

    x_train = data_inputx[:-testsize]
    x_test = data_inputx[-testsize:]

    # Split the targets into training/testing sets
    y_train = data_inputy[:-testsize]
    y_test = data_inputy[-testsize:] 

    return x_train, x_test, y_train, y_test
 
    
def scipy_linear_mod(x_train, x_test, y_train, y_test):
    #Source: Scikit Learn. (n.d). Linear Regression Example. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html 

    #regr = linear_model.LinearRegression()

    regr = linear_model.LogisticRegression() #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
 
    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    print(y_pred, ' y_pred')
 
    acc = accuracy_score(y_pred, y_test)

    print(acc, ' is accuracy_score') 

    auc = roc_auc_score(y_pred, y_test, average=None)

    print(auc, " AUC ")

    #metrics.plot_roc_curve(regr, y_pred, y_test)  
    #plt.savefig('auc.png')  
 


def main(): 
 
    x_train, x_test, y_train, y_test = read_data() # when you read from file 

    print(x_train, ' x_train')
    print(y_train, ' y_train') 


    scipy_linear_mod(x_train, x_test, y_train, y_test)
 


if __name__ == '__main__':
     main()
