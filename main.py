#This is a place for me to begin trying to understand neural networks and deep learning.. 
#This will be done through the use of youtube/google/stack overflow -- self taught

# TODO
# Learning Support Vector Machines 
# Linear Regression review 
# KNN algorithm
# Applying all of these fundementals in code and understanding them well
# Implement a neural network

import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plot 

#makes x y a numpy array, for faster calculations
#reshape to make x 2D 
x = np.array([5, 15, 25, 35, 45, 55]).reshape(-1, 1)
y = np.array([5, 20, 14, 32, 22, 38])

print(x)

#calcs the optimal weight values for x 
model = LinearRegression().fit(x, y)
        
#score() just determines the coefficient of determination
r_squared = model.score(x, y)

print("R^2 = ", r_squared)

print("Intercept = ", model.intercept_)
print("Slope = ", model.coef_)

y_pred = model.predict(x)
print("Prediction: :", y_pred)

#plot.plot(x,y)
#plot.show()

