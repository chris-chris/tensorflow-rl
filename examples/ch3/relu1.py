import numpy as np

from activation import ReLU

X = np.array([1.0, 1.0])
W = np.array([.5, 1.0])
b = 0.6

print("X :", X)
print("W :", W)
print("b :", b)

Y = np.dot(X, W) + b
print("Y = np.dot(X, W) + b")
print("Y :", Y)

Y = ReLU(Y)
print("Y = ReLU(Y)")
print("Y :", Y)