import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


df = pd.read_csv('D:\data.csv')
#df.head()
df_sort=df.sort_values(by=['output'],ascending=False)
P=[]
R=[]
tpr=[]
fpr=[]
last=0
lastlabel=0
sum=0
v=1
for index,rows in df_sort.iterrows():
    
    tp=0
    fp=0
    fn=0
    tn=0
    if v!=rows['output']:
        for index,row in df_sort.iterrows():
            if row['label']==1:
                if row['output']>=v:
                    tp+=1
                else:
                    fn+=1
            else:
                if row['output']>=v:
                    fp+=1
                else:
                    tn+=1
        if tp+fp==0:
            P.append(1)
            R.append(0)
            tpr.append(0)
            fpr.append(0)
        else:
            P.append(tp/(tp+fp))
            R.append(tp/(tp+fn))
            tpr.append(tp/(tp+fn))
            fpr.append(fp/(fp+tn))
        lastlabel=rows['label']
    else:
        if lastlabel!=rows['label']:
            lenth=len(tpr)
            tpr.append(tpr[lenth-1]+1)
            fpr.append(fpr[lenth-1]+1)
    v=rows['output']
    


for i in range(0,len(tpr)-2):
    sum+=(fpr[i+1]-fpr[i])*(tpr[i]+tpr[i+1])/2
sum+=(1-fpr[len(tpr)-1])*tpr[i]
print(sum)

plt.plot(R, P)
plt.figure(num="PR+ROC", figsize=(6, 4), facecolor="white", edgecolor="black")
plt.plot(fpr,tpr)
plt.show()

