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

Y = np.array([0,1,1,1])

epoch = 0
sum_error = 1
while sum_error != 0:
  epoch += 1
  print("learning epoch :", epoch)
  print("X :", X)
  output = perceptron(X, W)
  print("perceptron(X) :", output)
  errors = Y - output
  print("errors :", errors)
  W = update_weight(X, W, Y, errors)
  sum_error = np.sum(errors)
  print("sum of errors :", sum_error)
  print("W :", W)
  print("")

print("final W for OR function :", W)

"""
learning epoch : 6
X : [[ 1.  0.  0.]
 [ 1.  1.  0.]
 [ 1.  0.  1.]
 [ 1.  1.  1.]]
np.dot(X, W) : [-0.1  0.1  0.1  0.3]
perceptron(X) : [0 1 1 1]
errors : [0 0 0 0]
sum of errors : 0
W : [-0.1  0.2  0.2]

final W for OR function : [-0.1  0.2  0.2]
"""
