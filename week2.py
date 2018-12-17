import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt   
import matplotlib.pyplot as plt 
from scipy.special import expit #
from scipy import optimize

def sigmoid(x):
    return 1/(1+np.exp(-x))

def transpose(x):
	return x.T
def log(a):
	return np.log(a)
def h(mytheta,myX):
    return expit(np.dot(myX,mytheta))

def computeCost(mytheta,myX,myy,mylambda): 
    term1 = np.dot(-np.array(myy).T,np.log(h(mytheta,myX)))
    term2 = np.dot((1-np.array(myy)).T,np.log(1-h(mytheta,myX)))
    regterm = (mylambda/2) * np.sum(np.dot(mytheta[1:].T,mytheta[1:])) 
    return float( (1./m) * ( np.sum(term1 - term2) + regterm ) )
def optimizeTheta(mytheta,myX,myy,mylambda):
    result = optimize.fmin(computeCost, x0=mytheta, args=(myX, myy, mylambda), maxiter=400, full_output=True)
    return result[0], result[1]
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta) - y
    term = np.multiply(error, X)
    grad= term / len(X)
    ju=grad.sum(axis=0)
    return ju   
data = pd.read_csv('ex2data1.txt',header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]
data.head()

mask = y ==1
adm = plt.scatter(X[mask][0].values,X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values,X[~mask][1].values)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend((adm,not_adm),('Admitted','Not admitted'))
plt.show()

Xold=X
yold=y
(m,n)=X.shape
X=np.hstack((np.ones((m,1)), X))
y=y[:,np.newaxis]
theta = np.zeros((n+1,1)) 

J=computeCost(theta,X,y,0)
print(J)

print(gradient(theta,X,y))

theta, mincost = optimizeTheta(theta,X,y,0)


mask = yold ==1
adm = plt.scatter(Xold[mask][0].values,Xold[mask][1].values)
not_adm = plt.scatter(Xold[~mask][0].values,Xold[~mask][1].values)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend((adm,not_adm),('Admitted','Not admitted'))
boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
plt.show()
