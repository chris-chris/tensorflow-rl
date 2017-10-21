import numpy as np

def step(x):
  return np.where(x > 0, 1, 0)

def perceptron(X, W):
  h = np.dot(X, W)
  print("np.dot(X, W) :", h)

  return step(h)

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
while sum_error != 0:
  epoch += 1
  print("learning epoch :", epoch)
  print("X :", X)
  y = perceptron(X, W)
  print("perceptron(X) :", y)
  errors = Y - y
  print("errors :", errors)
  W = update_weight(X, W, Y, errors)
  sum_error = np.sum(errors)
  print("sum of errors :", sum_error)
  print("W :", W)
  print("")

print("final W for AND function :", W)

"""
learning epoch : 4
X : [[ 1.  0.  0.]
 [ 1.  1.  0.]
 [ 1.  0.  1.]
 [ 1.  1.  1.]]
np.dot(X, W) : [ -1.00000000e-01  -2.77555756e-17  -2.77555756e-17   1.00000000e-01]
perceptron(X) : [0 0 0 1]
errors : [0 0 0 0]
sum of errors : 0
W : [-0.1  0.1  0.1]

final W for AND function : [-0.1  0.1  0.1]
"""