import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

tp=0
fp=0
fn=0
tn=0

m_t=0
m_f=0

t_feature = np.mat(np.loadtxt(open('train_feature.csv','rb'),delimiter=',',skiprows=1))
t_feature  = np.c_[t_feature,np.ones(t_feature.shape[0])]
t_target=np.loadtxt(open('train_target.csv','rb'),delimiter=',',skiprows=1) 
beta=np.dot(np.dot(np.dot(t_feature.T,t_feature).I,t_feature .T),t_target).T

v_target=np.loadtxt(open('val_target.csv','rb'),delimiter=',',skiprows=1) 
v_feature=np.mat(np.loadtxt(open('val_feature.csv','rb'),delimiter=',',skiprows=1))
v_feature = np.c_[v_feature,np.ones(v_feature.shape[0])]
prediction1=np.dot(v_feature,beta)


for rows in range(len(t_target)): 
    if t_target[rows]==1:
        m_t+=1
    else:
        m_f+=1

for rows in range(len(v_target)): 
    tep=prediction1[rows].tolist()
    tempval=1/(1+pow(np.e,-tep[0][0]))
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
threshold=(m_t/m_f)/(1+m_t/m_f)

print('precision:', precision)
print('P:', P)
print('R:', R)
print('threshold:', threshold)


test_feature=np.mat(np.loadtxt(open('test_feature.csv','rb'),delimiter=',',skiprows=1))
test_feature = np.c_[test_feature,np.ones(test_feature.shape[0])]
prediction2=np.dot(v_feature,beta).tolist()
for rows in range(len(prediction2)): 
    tempval=1/(1+pow(np.e,-prediction2[rows][0]))
    if tempval>threshold:
        prediction2[rows][0]=1
    else:
        prediction2[rows][0]=0

prediction2.insert(0,[1])
np.savetxt('171860027_0.csv', prediction2, delimiter = ',') 
os.system("pause")