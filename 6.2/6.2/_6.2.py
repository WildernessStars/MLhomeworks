
from __future__ import print_function
import numpy as np
import pandas as pd
import operator
from sklearn import tree
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold  
class RandomForest():
   
    #T: int,轮数.
    def __init__(self, T):
        self.T = T


   
    # 对测试集进行预测
    def predict(self, X_test, X_train, y_train):
        modellist=[]
        featurelist=[]
        y_predlist=[]
        for i in range(self.T):
            xlist=[]
            ylist=[]
            featurelist.append(random.sample(range(0,14), 10))
            for j in range(len(y_train)):
                index=random.randint(0,len(y_train)-1)
                xlist.append(X_train[index,featurelist[i]])
                ylist.append(y_train[index])
            modellist.append(tree.DecisionTreeRegressor())
            modellist[i].fit(xlist, ylist)
            y_pred = modellist[i].predict(X_test[:,featurelist[i]])
            y_predlist.append(y_pred)
            
        y_pred=np.zeros(len(X_test))
        for i in range(self.T):
            for j in range(len(X_test)):
                y_pred[j]+=y_predlist[i][j]
        for j in range(len(X_test)):
            y_pred[j]=1 if y_pred[j]>self.T/2 else 0
        return y_pred
  

def main():
    clf = RandomForest(T=50)
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