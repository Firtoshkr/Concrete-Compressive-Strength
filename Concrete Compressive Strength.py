#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = np.array(pd.read_csv("C:\\Users\\firto\\Concrete_data.csv"))


# In[3]:


y = data[:,-1]


# In[4]:


X = data[:,:-1]


# In[5]:


print(X.shape,y.shape)3699*6+6896+96+96
print(X.shape,y.shape)


# In[6]:


u = np.mean(X)
std = np.std(X)
X = (X-u)/std


# In[7]:


ones = np.ones((1030,1))
X = np.hstack((ones,X))


# In[18]:


def hypothesis(X,theta):
    y_ = np.dot(X,theta)
    return y_

def error(X,y,theta):
    m,n = X.shape
    y_ = hypothesis(X,theta)
    err = np.sum((y_-y)**2)
    return err/m

def gradient(X,y,theta):
    m,n = X.shape
    y_ = hypothesis(X,theta)
    grad = np.zeros((n,))
    grad = np.dot((y_-y).T,X)
    return grad/m

def gradientDescent(X, y, learning_rate = 0.1, epoch = 300):
    m,n = X.shape
    err = []
    grad = np.zeros((n,))
    theta = np.zeros((n,))
    for i in range(epoch):
        er = error(X,y,theta)
        err.append(er)
        grad = gradient(X,y,theta)
        theta = theta - learning_rate * grad
    return err , theta


# In[19]:


err, theta = gradientDescent(X,y)
err


# In[10]:


plt.plot(err)


# In[11]:


def r2_score(y,ypred):
    ymean = y.mean()
    num = np.sum((y-ypred)**2)
    denum = np.sum((y-ymean)**2)
    score = 1 - num/denum
    return score


# In[12]:


ypred  = np.dot(X,theta)


# In[13]:


r2_score(y,ypred)

