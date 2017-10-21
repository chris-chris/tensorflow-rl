# Chapter1. Python, Numpy, Notebook

## 1-1. Python

Python은 머신러닝 업계의 사실상 표준으로 자리잡은 언어입니다. 현재 Python은 2.x 버전과 3.x 버전 두가지 버전으로 나뉘어져있습니다. 저는 이 튜토리얼에서 3.x 버전대 Python을 기준으로 튜토리얼을 작성할 것입니다. 하지만, 가능하면 2가지 버전에서 모두 호환이 되는 코드로 구현을 해보도록 하겠습니다.

시작부터, Numpy를 활용해서 행렬연산을 포함해 다양한 연산을 구현할 수 있는 기본 연산자들을 소개해드리도록 하겠습니다.


### Python 기본 연산

1) 더하기 연산입니다. x 와 y 값을 더하고 싶다면, `x + y` 이렇게 표현합니다.

```python
# 더하기(add)
x = 1
y = 2
z = x + y
print("x + y :", z)
# x + y : 3
```

2) 빼기 연산입니다. `x - y` 이렇게 표현합니다.

```python
# 빼기(subtract)
x = 3
y = 4
z = x - y
print("x - y :", z)
# x - y : -1
```

3) 곱하기 연산입니다. `x * y` 이렇게 표현합니다.

```python
# 곱하기(multiply)
x = 5
y = 4
z = x * y
print("x * y :", z)
# x * y : 20
```

4) 지수 연산입니다. `x ** y` 이렇게 표현합니다.

```python 
# 지수(exponential)
x = 2
y = 10
z = x ** y
print("x ** y :", z)
# x ** y : 1024
```

5) / 연산입니다. 나눗셈입니다. `x / y` 이렇게 표현합니다.

```python
# 나눗셈(division)
x = 7
y = 3
z = x / y
print("x / y :", z)
# x / y : 2.3333333333333335
```

6) % 연산입니다. 나눗셈을 한 후 나머지를 구하는 연산입니다.

```python
# 나머지(remainder)
x = 7
y = 3
z = x % y
print("x % y :", z)
# x % y : 2
```

7) // 연산입니다. 나눗셈을 한 후 소수점 이하를 버리는 연산입니다.

```python
# // 연산
x = 7
y = 3
z = x // y
print("x // y :", z)
# x // y : 2
```

8) += 연산입니다. 사실상 더하기 연산과 비슷합니다. 

x 값에 10을 더한 값을 x에 할당하고 싶을 때는 `x += 10` 이렇게 표현합니다.

```python
# += 연산자
x = 1
x += 10
print("x = 1")
print("x += 10")
print("x:", x)
# x: 11
```

9) *= 연산입니다. 사실상 곱하기 연산과 비슷합니다. 

x 값에 3을 곱한 값을 x에 할당하고 싶을 때는 `x *= 3` 이렇게 표현합니다.

```python
# *= 연산자
x = 2
x *= 3
print("x = 2")
print("x *= 3")
print("x :", x)
# x : 6
```

10) /= 연산입니다. 사실상 나누기 연산과 비슷합니다.

x 값이 3을 나눈 값을 x에 할당하고 싶을 때는 `x /= 3` 이렇게 표현합니다.

```
# /= 연산자
x = 7
x /= 3
print("x = 7")
print("x /= 3")
print("x :", x)
# x : 2.3333333333333335
```

11) type 함수. 변수의 자료형을 조회하는 함수입니다.

```python
# type 함수(type function)
x = 1
y = 1.5
print("x = 1")
print("type(x) :", type(x))
# type(x) : <class 'int'>

print("y = 1.5")
print("type(y) :", type(y))
# type(y) : <class 'float'>
```

### Python 기본 연산 전체 코드 (ch1/ops.py)

```python
# -*- encoding : utf-8 -*-
# -*- encoding : utf-8 -*-

x = 7
print("x = 7")
print("x :", x)

y = 3
print("y = 3")
print("y :", y)

# 더하기(add)
x = 7
y = 3
z = x + y
print("x + y :", z)
# x + y : 3

# 빼기(subtract)
x = 7
y = 3
z = x - y
print("x - y :", z)
# x - y : -1

# 곱하기(multiply)
x = 7
y = 3
z = x * y
print("x * y :", z)
# x * y : 20

# 지수(exponential)
x = 7
y = 3
z = x ** y
print("x ** y :", z)
# x ** y : 1024

# 나눗셈(division)
x = 7
y = 3
z = x / y
print("x / y :", z)
# x / y : 2.3333333333333335

# 나머지(remainder)
x = 7
y = 3
z = x % y
print("x % y :", z)
# x % y : 2

# // 연산
x = 7
y = 3
z = x // y
print("x // y :", z)
# x // y : 2

# += 연산자
x = 7
x += 10
print("x = 7")
print("x += 10")
print("x:", x)
# x: 11

# *= 연산자
x = 7
x *= 3
print("x = 7")
print("x *= 3")
print("x :", x)
# x : 6

# type 함수(type function)
x = 7
print("x = 7")
print("type(x) :", type(x))
# type(x) : <class 'int'>

y = 1.5
print("y = 1.5")
print("type(y) :", type(y))
# type(y) : <class 'float'>
```

## 1-2. Numpy


### Numpy 배열 연산 전체 코드 (ch1/numpy_ops.py)
```python
# -*- encoding : utf-8 -*-

import numpy as np

# add arrays
a = np.array([1,2,3])
b = np.array([2,3,4])

print("a = np.array([1,2,3])")
print("a :", a)
# a : [1 2 3]

print("b = np.array([2,3,4])")
print("b :",b)
# b : [2 3 4]

c = a + b

print("a + b :", c)
# a + b : [3 5 7]

c = a * b
print("a * b :", c)
# a * b : [ 2  6 12]

c = np.dot(a, b)
print("np.dot(a, b) :", c)
# np.dot(a,b) : 20

c = np.multiply(a, b)
print("np.multiply(a, b) : ", c)
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
```

## 1-3. Numpy 행렬 연산


### Numpy 행렬 연산 전체 코드 (ch1/matrix_ops.py)
```python
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
```