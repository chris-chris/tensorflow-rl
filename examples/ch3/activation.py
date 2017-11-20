import numpy as np

def sigmoid(x):
  return 1. / (1 + np.exp(-x))

def sigmoid_p(x):
  return sigmoid(x) * (1-sigmoid(x))

def tanh(x):
  return np.tanh(x)

def tanh_p(x):
  return 1. - x * x

def softmax(x):
  e = np.exp(x - np.max(x))
  if e.ndim == 1:
    return e / np.sum(e, axis=0)
  else:
    return e / np.array([np.sum(e, axis=1)]).T

def ReLU(x):
  return x * (x > 0)

def ReLU_p(x):
  return 1. * (x > 0)

def Perceptron(x):
  if(x > 0):
    return 1
  else:
    return 0