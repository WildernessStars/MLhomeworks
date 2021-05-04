from __future__ import print_function
import numpy as np
import pandas as pd
import operator
class KNN():
   
    #k: int,最近邻个数.
    def __init__(self, k=5):
        self.k = k

    # 此处需要填写，建议欧式距离，计算一个样本与训练集中所有样本的距离
    def distance(self, one_sample, X_train):
        sample_distance=[]
        for xtrain in X_train:
            sample_distance.append(np.sqrt(np.sum((one_sample - xtrain)**2)))
        return sample_distance
    
    # 此处需要填写，获取k个近邻的类别标签
    def get_k_neighbor_labels(self, distances, y_train, k):
        neighbor=[-1,-1,-1,-1,-1,-1,-1]
        for i in range(0,len(distances)):
            for j in range(0,k):
                if neighbor[j]==-1:
                    neighbor[j]=i
                    break
                elif distances[i]<=distances[neighbor[j]]:
                    neighbor.insert(j,i)
                    break
        neighbor_label=[]
        for index in range(0,k):
            neighbor_label.append(y_train[neighbor[index]])
        return neighbor_label
    # 此处需要填写，标签统计，票数最多的标签即该测试样本的预测标签
    def vote(self, one_sample, X_train, y_train, k):
        dis=self.distance(one_sample,X_train)
        neighbor_label=self.get_k_neighbor_labels(dis, y_train, k)
        votes={}
        for index in range(0,k):
            label=neighbor_label[index]
            if label in votes:
                votes[label]+=1
            else:
                votes[label]=1
        sorted_votes=sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]
    
    # 此处需要填写，对测试集进行预测
    def predict(self, X_test, X_train, y_train):
        y_test=[]
        for sample in X_test:
            y_test.append(self.vote(sample,X_train,y_train,self.k))
        return y_test
  

def main():
    clf = KNN(k=5)
    train_data = np.genfromtxt('./data/train_data.csv', delimiter=' ')
    train_labels = np.genfromtxt('./data/train_labels.csv', delimiter=' ')
    test_data = np.genfromtxt('./data/test_data.csv', delimiter=' ')
   
    #将预测值存入y_pred(list)内    
    y_pred = clf.predict(test_data, train_data, train_labels)
    np.savetxt("test_ypred.csv", y_pred, delimiter=' ')
  


if __name__ == "__main__":
    main()