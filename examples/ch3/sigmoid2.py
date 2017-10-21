import numpy as np

def sigmoid(x):
  return 1. / (1 + np.exp(-x))

def perceptron(X, W):
  h = np.dot(X, W)
  print("np.dot(X, W) :", h)

  return sigmoid(h)

def update_weight(X, W, Y, errors, learning_rate = 0.1):
  W = W + learning_rate * np.dot(errors, X)
  return W

W = np.array([0., 0., 0.])

X = np.array([[1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0]])

Y = np.array([0,0,0,1])

epoch = 0
sum_error = 1
threashold = 0.1
while sum_error > threashold:
  epoch += 1
  print("learning epoch :", epoch)
  print("X :", X)
  output = perceptron(X, W)
  print("perceptron(X) :", np.round(output, 4))
  errors = Y - output
  print("errors :", np.round(errors, 4))
  W = update_weight(X, W, Y, errors)
  sum_error = np.sum(np.absolute(errors)) / len(Y)
  print("sum of errors :", np.round(sum_error, 4))
  print("W :", np.round(W, 4))
  print("")

print("final W for AND function :", np.round(W, 4))

"""
learning epoch : 362
X : [[ 1.  0.  0.]
 [ 1.  1.  0.]
 [ 1.  0.  1.]
 [ 1.  1.  1.]]
np.dot(X, W) : [-5.68613928 -2.02976748 -2.02976748  1.62660431]
perceptron(X) : [ 0.0034  0.1161  0.1161  0.8357]
errors : [-0.0034 -0.1161 -0.1161  0.1643]
sum of errors : 0.1
W : [-5.6933  3.6612  3.6612]

final W for AND function : [-5.6933  3.6612  3.6612]
"""