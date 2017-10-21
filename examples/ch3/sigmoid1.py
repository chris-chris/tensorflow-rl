import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
  return 1. / (1 + np.exp(-x))

x = np.arange(-10., 10., 0.2)
y = sigmoid(x)

print("sigmoid(x) :", y)
plt.plot(x, y)
plt.show()