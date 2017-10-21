import numpy as np

A = np.array([[1,2,3],
              [4,5,6]])
print("A :")
print(A)

B = A.transpose()
print("B = A.transpose()")
print("B :")
print(B)
"""
A :
[[1 2 3]
 [4 5 6]]
B = A.transpose()
B :
[[1 4]
 [2 5]
 [3 6]]
"""