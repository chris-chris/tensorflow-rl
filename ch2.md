# Chapter2. Perceptron


## 2-1. 퍼셉트론(Perceptron)

퍼셉트론(Perceptron)이라는 알고리즘은 1957년 Frank Rosenblatt에 의해 발명되었습니다. 퍼셉트론은 인공지능 기술의 시작이라고 평가받고 있습니다. 퍼셉트론은 이미지 인식을 하기 위해 발명되었으며, 뉴론의 구조를 모방한 최초의 인공 뉴런입니다. 

![mark1](http://chris-chris.ai/img/ch2/mark1.jpeg)

![perceptron1](http://chris-chris.ai/img/ch2/perceptron1.png)

단일 퍼셉트론의 구조는 아래의 Python 코드로 표현할 수 있습니다. 퍼셉트론은 XW + b로 표현되는 선형 연산의 결과값을 구한 후에 Step 함수에 넣어 최종 결과를 얻습니다.

```python
import numpy as np

def step(x):
  return np.where(x > 0, 1, 0)

def perceptron(X):
  W = np.array([0.5, 0.5])
  b = 0
  h = np.dot(X, W) + b
  print("np.dot(X, W) + b :", h)

  return step(h)
```

먼저 Step 함수의 결과값은 어떻게 나오는 지 그래프로 확인해보겠습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

def step(x):
  return np.where(x > 0, 1, 0)

x = np.arange(-10., 10., 0.2)
y = step(x)

print("step(x) :", y)
plt.plot(x, y)
plt.show()
```

![step1](http://chris-chris.ai/img/ch2/step1.png)

AND 함수와 동일한 결과를 내놓는 퍼셉트론(Perceptron)을 구현해보겠습니다. AND는 x1와 x2가 모두 1일때의 결과가 1인 함수입니다. 우리는 AND 함수를 퍼셉트론으로 구현해볼 수 있습니다. 

우리가 원하는 입력값과 입력값에 해당하는 정답은 아래와 같습니다.

```
X = [0, 0] Y = 0
X = [1, 0] Y = 0
X = [0, 1] Y = 0
X = [1, 1] Y = 1
```

numpy로는 이렇게 표현해볼 수 있겠네요.

```python
X = np.array([[0.0, 0.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])

Y = np.array([0,0,0,1])
```

가중치 W와 바이어스 b의 초기값을 임의로 설정해보겠습니다. 가중치 W는 [0., 0.]로 설정하고, 편향 b는 0.으로 설정하겠습니다. 

```python
W = np.array([0., 0.])
b = 0
```

그런데, 계산을 단순하게 만들기 위해 편향을 제외할 방법이 있습니다. 바로 편향를 가중치의 첫번째 값으로 설정한 후, 무조건 X의 첫번째 입력값은 1로 고정시키는 것입니다. 그러면 Y = XW 행렬곱만으로 Y = XW + b 를 대체할 수 있습니다. 그럼, X, Y, W 의 값을 다시 정의해보도록 하겠습니다.

```python
W = np.array([0., 0., 0.])

X = np.array([[1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0]])

Y = np.array([0,0,0,1])
```

위 가중치 W와 편향 b을 가진 초기의 퍼셉트론에 4가지 입력값을 넣었을 때 각각 어떤 값을 반환하는 지 확인해보겠습니다.


```python
import numpy as np

def step(x):
  return np.where(x > 0, 1, 0)

def perceptron(X, W):
  h = np.dot(X, W)
  print("np.dot(X, W) :", h)

  return step(h)

def update_weight(X, W, Y, errors, learning_rate = 0.1):
  W = W + learning_rate * np.dot(errors, X)
  return W

W = np.array([0., 0., 0.])

X = np.array([[1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0]])

Y = np.array([0,0,0,1])

print("X :", X)
y = perceptron(X, W)
print("perceptron(X) :", y)
errors = Y - y
print("errors :", errors)
W = update_weight(X, W, Y, errors)
sum_error = np.sum(errors)
print("sum of errors :", sum_error)
print("W :", W)

"""
X : [[ 1.  0.  0.]
 [ 1.  1.  0.]
 [ 1.  0.  1.]
 [ 1.  1.  1.]]
np.dot(X, W) : [ 0.  0.  0.  0.]
perceptron(X) : [0 0 0 0]
errors : [0 0 0 1]
sum of errors : 1
W : [ 0.1  0.1  0.1]
"""
```

이렇게 결과값을 보면, 마지막 4번째 결과가 정답과 다른 것을 확인할 수 있습니다. 네번째 X값의 선형 연산 값을 확인해보면, 0으로 1보다 작았습니다. 그럼 이 선형 연산 값을 0보다 작거나 같게 만들려면 어떻게 해야 할까요? 

퍼셉트론을 구현하는 데는 성공했지만, 솔직히 가중치 W와 편향 b를 "학습시키는 법"에 대해서는 아직 잘 모르겠습니다. 물론 손으로 가중치를 임의로 변경해볼 수는 있습니다. 하지만, 우리는 우리의 손이 아닌 알고리즘으로 퍼셉트론을 학습시켜야 합니다. 우리가 원하는 답을 반환하도록 만들려면, 어떻게 만들어야 하는 걸까요? 인공 뉴런을 학습 시키는 법을 이어서 배워보도록 하겠습니다.


## 2-2. 퍼셉트론 학습시키기

퍼셉트론을 학습시키는 알고리즘을 구현해보도록 하겠습니다.

```python
import numpy as np

def step(x):
  return np.where(x > 0, 1, 0)

def perceptron(X, W):
  h = np.dot(X, W)
  print("np.dot(X, W) :", h)

  return step(h)

def update_weight(X, W, Y, errors, learning_rate = 0.1):
  W = W + learning_rate * np.dot(errors, X)
  return W

W = np.array([0., 0., 0.])

X = np.array([[1.0, 0.0, 0.0],
              [1.0, 1.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0]])

Y = np.array([0,0,0,1])

epoch = 0
sum_error = 1
while sum_error != 0:
  epoch += 1
  print("learning epoch :", epoch)
  print("X :", X)
  y = perceptron(X, W)
  print("perceptron(X) :", y)
  errors = Y - y
  print("errors :", errors)
  W = update_weight(X, W, Y, errors)
  sum_error = np.sum(errors)
  print("sum of errors :", sum_error)
  print("W :", W)
  print("")

print("final W for AND function :", W)

"""
learning epoch : 4
X : [[ 1.  0.  0.]
 [ 1.  1.  0.]
 [ 1.  0.  1.]
 [ 1.  1.  1.]]
np.dot(X, W) : [ -1.00000000e-01  -2.77555756e-17  -2.77555756e-17   1.00000000e-01]
perceptron(X) : [0 0 0 1]
errors : [0 0 0 0]
sum of errors : 0
W : [-0.1  0.1  0.1]

final W for AND function : [-0.1  0.1  0.1]
"""
```

이렇게 퍼셉트론에 학습 알고리즘을 적용하여, 퍼셉트론이 우리가 원하는 AND 연산과 동일한 결과를 반환하도록 구현했습니다. 


## 2-3. Sigmoid 인공 뉴런

퍼셉트론이 아닌 Sigmoid 인공 뉴런을 만들어보도록 하겠습니다. 아까 만들어본 퍼셉트론과의 가장 큰 차이점은 결과값이 0과 1로만 이루어지지 않는다는 점입니다. x값에 따른 Sigmoid 함수의 값을 한번 그래프로 그려보겠습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
  return 1. / (1 + np.exp(-x))

x = np.arange(-10., 10., 0.2)
y = sigmoid(x)

print("sigmoid(x) :", y)
plt.plot(x, y)
plt.show()
```

![sigmoid1](http://chris-chris.ai/img/ch2/sigmoid1.png)

이렇게 Sigmoid 함수는 Step 함수와는 다르게 값이 아주 부드럽게 나타납니다. Step 함수와는 다르게 이렇게 부드러운 결과가 왜 중요한 걸까요? 바로 우리가 배울 인공 뉴런의 학습 방법과 연관이 있습니다. 

우리는 조금씩 가중치를 조금씩 변경하면서 결과값을 수정해보겠습니다.  

```python
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

```python
def ReLU(x):
  return x * (x > 0)

def sigmoid(x):
  return 1/(1+np.exp(-x))
```

## 2-5. 손실 함수(Loss Function)



## 2-3. 경사 하강법(Gradient Descent)

```python
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

