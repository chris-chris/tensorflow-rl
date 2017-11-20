import matplotlib.pyplot as plt
import numpy as np

def step(x):
  return np.where(x > 0, 1, 0)

x = np.arange(-10., 10., 0.2)
y = step(x)

print("step(x) :", y)
plt.plot(x, y)
plt.show()