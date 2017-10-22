# -*- encoding : utf-8 -*-

import numpy as np
A = np.array([[1,2,3],
              [4,5,6]])
print("A :")
print(A)
"""
A :
 [[1 2 3]
 [4 5 6]]
"""

print("A :", A)
print("A.shape :", A.shape)
"""
A : [[1 2 3]
 [4 5 6]]
A.shape : (2, 3)
"""

# 첫번째 행을 선택
C = A[0][:]
print("C :", C)
print("C.shape :", C.shape)
"""
C : [1 2 3]
C.shape : (3,)
"""