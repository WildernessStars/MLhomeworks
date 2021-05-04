from __future__ import print_function
import numpy as np
import pandas as pd
import operator
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold  
class AdaBoost():
   
    #T: int,轮数.
    def __init__(self, T):
        self.T = T

    # 更新训练样本集的权值分布
    def update_weight(self, Dx, alpha,y_train, y_pred):
        for i in range(len(y_train)):
            Dx[i]=Dx[i]*np.exp(-alpha*y_train[i]*y_pred[i])
        return Dx/sum(Dx)

    # 计算弱分类器误差
    def error(self,Dx, y_train, y_pred):
        temp=0
        for i in range(len(y_train)):
            temp+=(Dx[i] if y_train[i]!=y_pred[i] else 0)
        return temp

   
   
    # 对测试集进行预测
    def predict(self, X_test, X_train, y_train):
        Dx=np.ones(len(y_train))/len(y_train)
        modellist=[]
        alphalist=[]
        early=self.T
        for i in range(self.T):
            modellist.append(tree.DecisionTreeRegressor())
            modellist[i].fit(X_train, y_train,sample_weight=Dx)
            y_pred = modellist[i].predict(X_train)
            
            error=self.error(Dx, y_train, y_pred)
            if error>0.5:
                early=i
                break;         
            alpha=0.5*np.log((1-error)/error)
            alphalist.append(alpha)
            Dx=self.update_weight(Dx, alpha,y_train, y_pred)
        y_pred=np.zeros(len(X_test))
        for i in range(early):
            temp=modellist[i].predict(X_test)
            for j in range(len(X_test)):
                y_pred[j]+=alphalist[i]*temp[j]
        for j in range(early):
            y_pred[j]=np.sign(y_pred[j])
        return y_pred
  

def main():
    clf = AdaBoost(T=60)
    train_data = np.genfromtxt('./adult_dataset/adult_train_feature.txt', delimiter=' ')
    train_labels = np.genfromtxt('./adult_dataset/adult_train_label.txt', delimiter=' ')
    test_data = np.genfromtxt('./adult_dataset/adult_test_feature.txt', delimiter=' ')
    test_labels = np.genfromtxt('./adult_dataset/adult_test_label.txt', delimiter=' ')
    
    '''
   五折交叉验证
    kfold = KFold(5, True, 10) 
    aucscores = [] 
    for train_index, test_index in kfold.split(train_data):
        y_pred = clf.predict(test_data, train_data[:,train_index], train_labels[:,train_index])
        aucscore = roc_auc_score(test_labels, y_pred)
        aucscores.append(aucscores) 
    print(np.mean(aucscores))
    '''
    #将预测值存入y_pred(list)内    
    y_pred = clf.predict(test_data, train_data, train_labels)
    acc = 0
    for i in range(len(y_pred)) :
        if y_pred[i] == test_labels[i] :
            acc += 1
    acc /= len(y_pred)
    print(roc_auc_score(test_labels, y_pred))
    np.savetxt("test_ypred.txt", y_pred, delimiter=' ')
  


if __name__ == "__main__":
    main()