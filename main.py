#This is a place for me to begin trying to understand neural networks and deep learning.. 
#This will be done through the use of youtube/google/stack overflow -- self taught

# TODO
# Learning Support Vector Machines 
# Linear Regression review 
# KNN algorithm
# Applying all of these fundementals in code and understanding them well
# Implement a neural network
import pandas as pd 
import numpy as np
<<<<<<< HEAD
from sklearn import linear_model
from sklearn.utils import shuffle 


data = pd.read_csv("student-mat.csv", sep = ";")

data = data[["Grade1", "Grade2", "Grade3", "Study-time", "Failures", "Absences"]]

predict = "Grade3"
=======
import csv 

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plot 
from numpy import genfromtxt
"""
#makes x y a numpy array, for faster calculations
#reshape to make x 2D 
x = np.array([5, 15, 25, 35, 45, 55]).reshape(-1, 1)
y = np.array([5, 20, 14, 32, 22, 38])
>>>>>>> d84d5cc9769d9107fcf3d0ba0cf8476cf90388e3

x = np.array(data.drop([predict], 1)
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
linear.score(x_test, y_test)

accuracy = linear.score(x_test, y_test)
print(accuracy)

<<<<<<< HEAD
print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)
=======
plot.plot(x,y)
plot.show()"""


with open("test.csv", 'r') as file:
    read = csv.reader(file)
    data = genfromtxt("test.csv", delimiter = ',')
    print(data[1])
>>>>>>> d84d5cc9769d9107fcf3d0ba0cf8476cf90388e3

