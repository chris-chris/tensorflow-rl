import numpy as np

def sigmoid(x):
  return 1. / (1 + np.exp(-x))

def sigmoid_p(x):
  return sigmoid(x) * (1-sigmoid(x))

def perceptron(X, W):
  h = np.dot(X, W)
  print("np.dot(X, W) :", h)

  return sigmoid(h)

def update_weight(X, W, Y, error_term, learning_rate = 0.1):
  W = W + learning_rate * np.dot(error_term, X)
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
  h = np.dot(X, W)
  output = sigmoid(h)
  print("perceptron(X) :", np.round(output, 4))
  errors = Y - output
  print("errors :", np.round(errors, 4))
  output_grad = sigmoid_p(h)
  print("output_grad :", np.round(output_grad, 4))
  error_term = errors * output_grad
  print("error_term :", np.round(error_term, 4))
  W = update_weight(X, W, Y, error_term)
  sum_error = np.sum(np.absolute(errors)) / len(Y)
  print("sum of errors :", np.round(sum_error, 4))
  print("W :", np.round(W, 4))
  print("")

print("final W for AND function :", np.round(W, 4))

"""

learning epoch : 2365
X : [[ 1.  0.  0.]
 [ 1.  1.  0.]
 [ 1.  0.  1.]
 [ 1.  1.  1.]]
perceptron(X) : [ 0.0035  0.1238  0.1238  0.8512]
errors : [-0.0035 -0.1238 -0.1238  0.1488]
output_grad : [ 0.0035  0.1085  0.1085  0.1266]
sum of errors : 0.1
W : [-5.6581  3.7012  3.7012]

final W for AND function : [-5.6581  3.7012  3.7012]
"""