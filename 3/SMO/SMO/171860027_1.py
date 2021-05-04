import os
import pandas as pd
import numpy as np

X_train=pd.read_csv('./X_train.csv',header=None).values
y_train=pd.read_csv('./y_train.csv',header=None).values

lenth = len(X_train)
for i in range(lenth):
    if y_train[i]==0:
        y_train[i]=-1
alpha=np.random.random_sample(lenth)
w=np.zeros(len(X_train[0]))
for i in range(lenth):
    w+=y_train[i]*alpha[i]*X_train[i]
b=np.mean(y_train.T[0] - np.dot(w, X_train.T))

def select_alpha_i(b_new):
    E=0;maxE=0
    for i in range(lenth):
        E=np.dot(w, X_train[i])+b_new-y_train[i][0]
        if maxE<abs(E):
            maxE=abs(E)
            index_i=i
            E1=E
    return E1,index_i


def select_alpha_j(E1,b_new):
    E=0;maxEdis=0#两个E差距最大
    for i in range(lenth):
        E=np.dot(w, X_train[i])+b_new-y_train[i][0]
        if maxEdis<abs(E1-E):
            maxEdis=abs(E1-E)
            index_j=i
            E2=E
    return E2,index_j

def calculate_alphanew(alpha_i,alpha_j,index_i,index_j,b,w,E1,E2):   
    alpha_j_old=alpha_j
    alpha_i_old=alpha_i
    if y_train[index_i]==y_train[index_j]:
        L=max(0,alpha_j+alpha_i-1)
        H=min(1,alpha_j+alpha_i)
    else:
        L=max(0,alpha_j-alpha_i)
        H=min(1,alpha_j-alpha_i+1)
    k11=np.dot(X_train[index_i],X_train[index_i])
    k12=np.dot(X_train[index_i],X_train[index_j])
    k22=np.dot(X_train[index_j],X_train[index_j])
    alpha_j=alpha_j+y_train[index_j]*(E1-E2)/(k11+k22-2*k12)
    if alpha_j>H:
        alpha_j=H
    elif alpha_j<L:
        alpha_j=L
    alpha_i=alpha_i+y_train[index_i]*y_train[index_j]*(alpha_j_old-alpha_j)
    alpha[index_i]=alpha_i
    alpha[index_j]=alpha_j
    w+=(alpha_i-alpha_i_old)*y_train[index_i]*X_train[index_i]+(alpha_j-alpha_j_old)*y_train[index_j]*X_train[index_j]
    b1new=b-E1-y_train[index_i]*k11*(alpha_i-alpha_i_old)-y_train[index_j]*k12*(alpha_j-alpha_j_old)
    b2new=b- E1-y_train[index_i]*k12*(alpha_i-alpha_i_old)-y_train[index_j]*k22*(alpha_j-alpha_j_old)
    return (b1new+b2new)/2,w

def smo(b,w):
    count=0
    while True:
        if count==2000:
            b = np.mean(y_train.T[0] - np.dot(w, X_train.T))
            print('w = ', w)
            print('b = ', b)
            break
        E1,index_i=select_alpha_i(b)
        E2,index_j=select_alpha_j(E1,b)
        b,w=calculate_alphanew(alpha[index_i],alpha[index_j],index_i,index_j,b,w,E1,E2)
        count+=1
    return w,b


def test(w,b,X,lenth):
    y = np.zeros(lenth)
    for i in range(lenth) :
        y[i]=np.dot(w, X[i])+b
        if y[i]>0:
            y[i]=1
        else:
            y[i]=0
    return y

w,b=smo(b,w)
y_train_test=test(w,b,X_train,lenth)
acc = 0
for i in range(lenth) :
    if y_train_test[i]==y_train[i] or y_train_test[i]==y_train[i]+1:
        acc+=1
print('acc = ', acc/lenth)

X_test=pd.read_csv('./X_test.csv',header=None).values
res = test(w,b,X_test,len(X_test))

np.savetxt('171860027_卫歆.csv', res, delimiter = ',') 

os.system("pause")


