import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# load and prepare the data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['benign'])
X.insert(0, 'x0', np.ones(X.shape[0]))

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42)

def g(x):
  return 1.0/(1.0+np.exp(-x))
g = np.vectorize(g)

def cost_function(y, h):
  J = np.sum(np.multiply(y, np.log(h)) + (1-y).multiply(np.log(1-h)))
  return -J/y.shape[0]


np.random.seed(42)
theta1 = np.random.randn(31, 30)
theta2 = np.random.randn(30, 1)
epsilon = 0.1 # learning rate

for i in range(100):
  # forward propagation
  z2 = np.dot(X_train, theta1)
  a2 = g(z2)
  z3 = np.dot(a2, theta2)
  h = g(z3)
  # J = cost_function(y_train, h)

  # back propagation
  a1 = X_train
  D3 = h - y_train
  D2 = np.dot(D3, theta2.transpose())

  dtheta2 = np.dot(a2.transpose(), D3)
  dtheta1 = np.dot(a1.transpose(), D2)

  theta1 = theta1 + dtheta1*epsilon
  theta2 = theta2 + dtheta2*epsilon

  if i%10==0:
    print 'dtheta1', dtheta1
    print 'dtheta2', dtheta2
    print 'total', (y_train==h).sum()


# print cost_function(y_train, h)
print 'total', (y_train==h).sum()











