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
from sklearn import linear_model
from sklearn.utils import shuffle 


data = pd.read_csv("student-mat.csv", sep = ";")

data = data[["Grade1", "Grade2", "Grade3", "Study-time", "Failures", "Absences"]]

predict = "Grade3"

x = np.array(data.drop([predict], 1)
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
linear.score(x_test, y_test)

accuracy = linear.score(x_test, y_test)
print(accuracy)

print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)

