import numpy as np
import pandas as pd
import operator
import csv
def main():
    trainset=pd.read_csv('train.csv',header=None,sep=',')
    dict = {}#记录Nextport为空，且暂不确定的订单GPS数据
    currenttrip = 0#记录订单当前的Nextport
    train_gps_index = 0#记录订单索引
    for train_gps_index,row in trainset.iterrows():
        loadingOrder=row[0]
        vesselNextport=row[8]   
        speed=row[6]
        if speed==0:          
            if loadingOrder in dict:
                trainset[8][train_gps_index]=dict[loadingOrder]
            else:
                trainset[8][train_gps_index]=currenttrip
            dict[loadingOrder]=currenttrip
            currenttrip+=1                
        else:
            if loadingOrder in dict:
                trainset[8][train_gps_index]=dict[loadingOrder]
            else:
                trainset[8][train_gps_index]=currenttrip
                dict[loadingOrder]=currenttrip
                currenttrip+=1

            
          
 

    #print(trainset)
    trainset.to_csv('train2.csv',sep=',',index=False,header=False)
if __name__ == "__main__":
    main()