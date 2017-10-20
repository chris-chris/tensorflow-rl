# Chapter2. Perceptron


## 2-1. 퍼셉트론(Perceptron)

AND, OR을 구현하는 퍼셉트론(Perceptron)을 구현해보겠습니다. AND는 x1와 x2가 모두 1일때의 결과가 1인 함수입니다. 우리는 AND 함수를 퍼셉트론으로 구현해볼 수 있습니다. 

단일 퍼셉트론의 구조는 아래의 Python 코드로 표현할 수 있습니다.

```
import numpy as np
def Perceptron(X):
  W = np.array([0.5, 0.5])
  b = 0.5
  h = np.dot(X, W) + b
  print("np.dot(X, W) + b :", h)
  if(h > 0):
    return 1
  else:
    return 0
```

우리가 원하는 입력값과 입력값에 해당하는 정답은 아래와 같습니다.

```
X = [0, 0] Y = 0
X = [1, 0] Y = 0
X = [0, 1] Y = 0
X = [1, 1] Y = 1
```

가중치 W와 바이어스 b의 초기값을 임의로 설정해보겠습니다. 가중치 W는 [0.5, 0.5]로 설정하고, 편향 b는 0으로 설정하겠습니다.

```
W = np.array([0.5, 0.5])
b = 0
```

위 가중치 W와 편향 b을 가진 초기의 퍼셉트론에 4가지 입력값을 넣었을 때 각각 어떤 값을 반환하는 지 확인해보겠습니다.


```python
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
```

이렇게 결과값을 보면, X2와 X3가 정답과 다른 것을 확인할 수 있습니다. X2와 X3의 값의 선형 연산 값을 확인해보면, 둘 다 0.5로 0보다 컸습니다. 그럼 이 선형 연산 값을 0보다 작거나 같게 만들려면 어떻게 해야 할까요? 아주 간단한 방법으로는, 편향(b) 값을 -0.5 이하이고 -1.0 보다 큰 값으로 설정하는 것입니다.

$$ b = (-1.0, -0.5]$$

예를 들어, 편향(b) 값을 -0.7로 설정하면, 무조건 모든 선형 연산에 -0.7을 더하기 때문에 오직 [1.0, 1.0]의 값을 가진 X4만 선형 연산 결과 값이 0보다 크게 될 것입니다.



```python
import numpy as np

def Perceptron(X):
  W = np.array([0.5, 0.5])
  b = -0.7
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
# np.dot(X, W) + b : -0.7
# Perceptron(X1) : 0

print("X2 :", X2)
print("Perceptron(X2) :", Perceptron(X2))
# X2 : [ 1.  0.]
# np.dot(X, W) + b : -0.2
# Perceptron(X2) : 0

print("X3 :", X3)
print("Perceptron(X3) :", Perceptron(X3))
# X3 : [ 0.  1.]
# np.dot(X, W) + b : -0.2
# Perceptron(X3) : 0

print("X4 :", X4)
print("Perceptron(X4) :", Perceptron(X4))
# X4 : [ 1.  1.]
# np.dot(X, W) + b : 0.3
# Perceptron(X4) : 1
```

이렇게 편향 값을 조정해서, 퍼셉트론이 우리가 원하는 AND 연산과 동일한 결과를 반환하도록 구현했습니다. 


```python
import numpy as np
X = np.array([1,1])
```

## 2-2. 인공 뉴런 : Sigmoid

```

def sigmoid(x):
  return 1. / (1 + np.exp(-x))

def sigmoid_p(x):
  return sigmoid(x) * (1-sigmoid(x))
  
```

## 3 X 3 사이즈의 이미지 인식 구현

3 X 3 사이즈를 가진 그림으로부터 0, /,  \, X 4가지 기호를 구분하는 Multi-Level Perceptron 을 구현해보겠습니다. 

4가지 답안 O, /, \, X 을 One-hot Encoding으로 표현해보면 아래와 같아집니다.

O = [1, 0, 0, 0]

/ = [0, 1, 0, 0]

\ = [0, 0, 1, 0]

X = [0, 0, 0, 1]


## 2-4. 활성화 함수(Activation Function)

```
def ReLU(x):
  return x * (x > 0)

def sigmoid(x):
  return 1/(1+np.exp(-x))
```

## 2-5. 손실 함수(Loss Function)



## 2-3. 경사 하강법(Gradient Descent)

```
# From udacity Machine Learning Nanodegree course

import numpy as np

# Define sigmoid function
def sigmoid(x):
  return 1/(1+np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
  return sigmoid(x) * (1 - sigmoid(x))

# Feature data
feature = np.array([0.9, -0.2])

# Label data (Target)
label = 0.9

# Weights of neural network
weights = np.array([0.3, -0.8])

# The learning rate, eta in the weight step equation
learnrate = 0.1

# the linear combination performed by the node (h in f(h) and f'(h))
h = np.dot(feature, weights)

# The neural network output (label-hat)
nn_output = sigmoid(h)

# output error (label - label-hat)
error = label - nn_output

# output gradient (f'(h))
output_grad = sigmoid_derivative(h)

# error term (lowercase delta)
error_term = error * output_grad

# Gradient descent step 
del_w = learnrate * error_term * feature

print('Output: %s' % nn_output)
print('Error: %s' % error)
print('Change in Weights: %s' % del_w)
```


## 2-5. 최적화 함수(Optimization Function)

