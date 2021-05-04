import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math

t_feature = np.mat(np.loadtxt(open('train_feature.csv','rb'),delimiter=',',skiprows=1))
t_feature  = np.c_[t_feature,np.ones(t_feature.shape[0])]
t_target=np.loadtxt(open('train_target.csv','rb'),delimiter=',',skiprows=1) 
beta=np.mat([0,0,0,0,0,0,0,0,0,0,0])

p=0
fd_twofrom=1
last_fd_twofrom=2
while last_fd_twofrom-fd_twofrom>=10e-13:
    sum=0
    xb=np.dot(t_feature,beta.T)
    for i in range(t_feature.shape[0]):
        temp=np.dot(beta,t_feature[i].T).tolist()
        sum+=-t_target[i]*temp[0][0]+math.log(1+pow(np.e,temp[0][0]))
    fd=0
    sd=0
    for i in range(xb.shape[0]):  
        p=pow(np.e,xb[0].tolist()[0][0])
        p=p/(1+p)
        fd-=(t_target[i]-p)*t_feature[i].T
        sd+=np.dot(t_feature[i].T,t_feature[i])*p*(1-p)
    last_fd_twofrom=fd_twofrom
    fd_twofrom=np.linalg.norm(fd)
    beta=beta-(np.dot(sd.I,fd)).T



v_target=np.loadtxt(open('val_target.csv','rb'),delimiter=',',skiprows=1) 
v_feature=np.mat(np.loadtxt(open('val_feature.csv','rb'),delimiter=',',skiprows=1))
v_feature = np.c_[v_feature,np.ones(v_feature.shape[0])]
prediction1=np.dot(v_feature,beta.T).tolist()

tp=0
fp=0
fn=0
tn=0

for rows in range(len(v_target)): 
    tempval=1/(1+pow(np.e,-prediction1[rows][0]))
    if v_target[rows]==1:
        if tempval>0.5:
             tp+=1
        else:
             fn+=1
    else:
        if tempval>0.5:
             fp+=1
        else:
             tn+=1

P=(tp/(tp+fp))
R=(tp/(tp+fn))
precision=(tp+tn)/(tp+fn+fp+tn)

print('precision:', precision)
print('P:', P)
print('R:', R)

test_feature=np.mat(np.loadtxt(open('test_feature.csv','rb'),delimiter=',',skiprows=1))
test_feature = np.c_[test_feature,np.ones(test_feature.shape[0])]
prediction2=np.dot(v_feature,beta.T).tolist()
for rows in range(len(prediction2)): 
    tempval=1/(1+pow(np.e,-prediction2[rows][0]))
    if tempval>0.5083333:
        prediction2[rows][0]=1
    else:
        prediction2[rows][0]=0

prediction2.insert(0,[1])
np.savetxt('171860027_1.csv', prediction2, delimiter = ',') 
os.system("pause")