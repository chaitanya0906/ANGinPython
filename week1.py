import sys
import json
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def computeCost(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)

def gradientDescent(X,y,theta,alpha,iteration):
	for i in range(iteration):
		temp= np.dot(X,theta)-y
		temp=np.dot(X.T,temp)
		theta = theta -(alpha/m)*temp
	return theta  
data = pd.read_csv('ex1data1.txt', header = None)
#reading from dataset

X= data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m=len(y)
data.head()

plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

X=X[:,np.newaxis] #code to convert to 2 Ranked array
y=y[:,np.newaxis] #code to convert to 2 Ranked array
theta= np.zeros([2,1])
iteration=1500
alpha=0.01
ones=np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term

J = computeCost(X, y, theta)	
print(J)

theta= gradientDescent(X,y,theta,alpha,iteration)

print(theta)

J = computeCost(X, y, theta)	
print(J)

plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()

