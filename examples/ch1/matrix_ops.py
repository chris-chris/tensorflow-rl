# -*- encoding : utf-8 -*-

import numpy as np

# add arrays
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

B = np.array([[2,3,4],
              [5,6,7],
              [8,9,10]])

print("A :\n", A)
"""
A :
 [[1 2 3]
 [4 5 6]
 [7 8 9]]
"""

print("B :\n",B)
"""
B :
 [[ 2  3  4]
 [ 5  6  7]
 [ 8  9 10]]
"""

C = A + B

print("A + B :\n", C)
"""
A + B :
 [[ 3  5  7]
 [ 9 11 13]
 [15 17 19]]
"""

C = A * B
print("A * B :\n", C)
"""
A * B :
 [[ 2  6 12]
 [20 30 42]
 [56 72 90]]
"""

C = np.dot(A, B)
print("np.dot(A, B) :\n", C)
"""
np.dot(A, B) :
 [[ 36  42  48]
 [ 81  96 111]
 [126 150 174]]
"""

C = np.multiply(A, B)
print("np.multiply(A, B) :\n", C)
"""
np.multiply(A, B) :
 [[ 2  6 12]
 [20 30 42]
 [56 72 90]]
"""

C = A.shape
print("A.shape :", C)
# A.shape : (3, 3)

C = A.ndim
print("A.ndim :", C)
# A.ndim : 2

C = A.dtype.name
print("A.dtype.name :", C)
# A.dtype.name : int64

C = A.size
print("A.size :", C)
# A.size : 9

C = type(A)
print("type(A) :", C)
# type(A) : <class 'numpy.ndarray'>