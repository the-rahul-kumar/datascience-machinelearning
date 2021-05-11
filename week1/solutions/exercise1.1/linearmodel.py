import numpy as np 

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score


import random


def get_data():
    #Source: University of California. (n.d). Machine-learning-databases. http://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    #Source: University of California. (n.d). Machine learning repository. http://archive.ics.uci.edu/ml/datasets/iris 
    #Source: Iris flower dataset. (2020). Wikipedia. https://en.wikipedia.org/wiki/Iris_flower_data_set
 
    #house_data = datasets.load_boston() #Scikit-learn provides a handy description of the dataset, and it can be easily viewed by:
  

    data_in = datasets.load_iris() 
 
    data_input = data_in.data[:, 1] # 1 feature  

    #data_input = data_in.data[:, [1, 2]] # 2 features 

    #data_input = data_in.data #when you want all features

    x_train = data_input[:-20]
    x_test = data_input[-20:]

    # Split the targets into training/testing sets
    y_train = data_in.target[:-20]
    y_test = data_in.target[-20:] 
    return x_train, x_test, y_train, y_test

 

def read_data():
    #Source: University of California. (n.d). Machine-learning-databases. http://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    #Source: University of California. (n.d). Machine learning repository. http://archive.ics.uci.edu/ml/datasets/iris 
    #Source: Iris flower dataset. (2020). Wikipedia. https://en.wikipedia.org/wiki/Iris_flower_data_set


    #data_in = genfromtxt("irisprocessed.csv", delimiter=",")

    data_in = genfromtxt("irisbinary.csv", delimiter=",")



    data_inputx = data_in[:,0:4] # all features 0, 1, 2, 3

    #data_inputx = data_in[:,[1]]  # one feature

    #data_inputx = data_in[:,[1,2]]  # two features

    data_inputy = data_in[:,-1] # this is target - so that last col is selected from data

    x_train = data_inputx[:-20]
    x_test = data_inputx[-20:]

    # Split the targets into training/testing sets
    y_train = data_inputy[:-20]
    y_test = data_inputy[-20:] 

    return x_train, x_test, y_train, y_test
 
    
def scipy_linear_mod(x_train, x_test, y_train, y_test):
    #Source: Scikit Learn. (n.d). Linear Regression Example. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html 
 
    print(' running scipy linear model')

    regr = linear_model.LinearRegression()


    # Create linear regression object

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    print(y_pred, ' y_pred')

    #y_pred_int = y_pred.astype(int) # n case you want to just convert to int

    y_pred_int = np.rint(y_pred) # round off and convert to int

    print(y_pred_int, ' y_pred int ')

    print(y_test, ' y_test')

    acc = accuracy_score(y_pred_int, y_test)

    print(acc, ' is accuracy_score')


    # Plot outputs if univariate case
    '''plt.scatter(x_test, y_test,  color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.savefig('resultlinear_reg.png')'''




def main(): 

    #x_train, x_test, y_train, y_test = get_data() # when you read from scikit-learn
 
    x_train, x_test, y_train, y_test = read_data() # when you read from file 

    print(x_train, ' x_train')
    print(y_train, ' y_train') 


    scipy_linear_mod(x_train, x_test, y_train, y_test)
 


if __name__ == '__main__':
     main()
