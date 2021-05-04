import os
import pandas as pd
import numpy as np
from cvxopt import base,solvers

X_train=pd.read_csv('./X_train.csv',header=None).values
y_train=pd.read_csv('./y_train.csv',header=None).values

lenth = len(X_train)
for i in range(lenth):
    if y_train[i][0]==0:
        y_train[i][0]=-1

P=base.matrix(np.zeros((lenth,lenth)))
q=base.matrix(np.zeros((lenth,1)))
A=np.zeros(lenth)
for i in range(lenth):
    for j in range(lenth):
        P[i,j]=y_train[i][0]*y_train[j][0]*np.dot(X_train[i].T,X_train[j])


for i in range(lenth):
    q[i,0]=-1.
    A[i]=y_train[i][0]

A=base.matrix(A)

b=base.matrix(0.)
G=base.matrix(np.zeros((lenth,lenth)))
for i in range(lenth):
    G[i,i]=-1.
h=base.matrix(np.zeros((lenth,1)))
for i in range(lenth):
    h[i,0]=0.
sv=solvers.qp(P,q,G,h,A.T,b)
alpha=sv['x']
print(sv['x'])
w=np.zeros(len(X_train[0]))
for i in range(lenth):
    w+=y_train[i][0]*alpha[i]*X_train[i]
b=np.mean(y_train.T[0] - np.dot(w, X_train.T))
print('w = ', w)
print('b = ', b)
def test(w,b,X,lenth):
    y = np.zeros(lenth)
    for i in range(lenth) :
        y[i]=np.dot(w, X[i])+b
        if y[i]>0:
            y[i]=1
        else:
            y[i]=0
    return y
y_train_test=test(w,b,X_train,lenth)
os.system("pause")