import numpy as np
import pandas as pd
from numpy import *
from pandas import *
import matplotlib.pyplot as plt

x = np.array([[1, 2], [2, 1], [3, 2.5], [4, 3],
              [5, 4], [6, 5], [7, 2.7], [8, 4.5],
              [9, 2]])

x = np.array([[175.0, 16.0, 64.0, 1.0,28776.56], [70.0, 2.0, 24.0, 1.0,12018.0],
              [54.0, 1.0, 8.0, 1.0,29866.0], [73.0, 5.0, 7.0, 1.0,12237.0]])

m, n = np.shape(x)
x_data = np.ones((m, n))
x_data[:, :-1] = x[:, :-1]
y_data = x[:, -1]

print (x_data.shape)
print (y_data.shape)
m, n = np.shape(x_data)
theta = np.ones(n)


def batchGradientDescent(maxiter, x, y, theta, alpha):
    xTrains = x.transpose()
    for i in range(0, maxiter):
        hypothesis = np.dot(x, theta)
        loss = (hypothesis - y)
        print (loss)
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
        cost = 1.0 / 2 * m * np.sum(np.square(np.dot(x, np.transpose(theta)) - y))
        #print ("cost: %f" % cost)
    return theta


result = batchGradientDescent(100, x_data, y_data, theta, 0.00001)
newy = np.dot(x_data, result)
fig, ax = plt.subplots()
ax.plot(x[:, 0], newy, 'k--')
ax.plot(x[:, 0], x[:, 1], 'ro')
plt.show()
print ("final: ", result)