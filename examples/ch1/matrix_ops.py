# -*- encoding : utf-8 -*-

import numpy as np

# add arrays
a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

b = np.array([[2,3,4],
              [5,6,7],
              [8,9,10]])

print("a :\n", a)
# a : [1 2 3]

print("b :\n",b)
# b : [2 3 4]

c = a + b

print("a + b :\n", c)
# a + b : [3 5 7]

c = a * b
print("a * b :\n", c)
# a * b : [ 2  6 12]

c = np.dot(a, b)
print("np.dot(a, b) :\n", c)
# np.dot(a,b) : 20

c = np.multiply(a, b)
print("np.multiply(a, b) :\n", c)
# np.multiply(a,b) : [ 2  6 12]

c = a.shape
print("a.shape :", c)
# a.shape : (3,)

c = a.ndim
print("a.ndim :", c)
# np.ndim(a) : 1

c = a.dtype.name
print("a.dtype.name :", c)
# a.dtype.name : int64

c = a.size
print("a.size :", c)
# a.size : 3

c = type(a)
print("type(a) :", c)
# type(a) : <class 'numpy.ndarray'>