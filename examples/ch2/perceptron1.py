import numpy as np

def Perceptron(X):
  W = np.array([0.5, 0.5])
  b = 0
  h = np.dot(X, W) + b
  print("np.dot(X, W) + b :", h)
  if(h > 0):
    return 1
  else:
    return 0

X1 = np.array([0.0, 0.0])
X2 = np.array([1.0, 0.0])
X3 = np.array([0.0, 1.0])
X4 = np.array([1.0, 1.0])

print("X1 :", X1)
print("Perceptron(X1) :", Perceptron(X1))
# X1 : [ 0.  0.]
# np.dot(X, W) + b : 0.0
# Perceptron(X1) : 0

print("X2 :", X2)
print("Perceptron(X2) :", Perceptron(X2))
# X2 : [ 1.  0.]
# np.dot(X, W) + b : 0.5
# Perceptron(X2) : 1

print("X3 :", X3)
print("Perceptron(X3) :", Perceptron(X3))
# X3 : [ 0.  1.]
# np.dot(X, W) + b : 0.5
# Perceptron(X3) : 1

print("X4 :", X4)
print("Perceptron(X4) :", Perceptron(X4))
# X4 : [ 1.  1.]
# np.dot(X, W) + b : 1.0
# Perceptron(X4) : 1