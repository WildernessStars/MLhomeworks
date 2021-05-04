import numpy as np
from dtw import *
import pandas as pd

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})


def TimeSeriesSimilarityImprove(s1, s2):
    # 取较大的标准差
    sdt = np.std(s1, ddof=1) if np.std(s1, ddof=1) > np.std(s2, ddof=1) else np.std(s2, ddof=1)
    # print("两个序列最大标准差:" + str(sdt))
    l1 = len(s1)
    l2 = len(s2)
    paths = np.full((l1 + 1, l2 + 1), np.inf)  # 全部赋予无穷大
    sub_matrix = np.full((l1, l2), 0)  # 全部赋予0
    max_sub_len = 0

    paths[0, 0] = 0
    for i in range(l1):
        for j in range(l2):
            d = np.linalg.norm(s1[i] - s2[j], ord=2)
            cost = d ** 2
            paths[i + 1, j + 1] = cost + min(paths[i, j + 1], paths[i + 1, j], paths[i, j])
            if np.linalg.norm(s1[i] - s2[j], ord=2) < sdt/10:
                if i == 0 or j == 0:
                    sub_matrix[i][j] = 1
                else:
                    sub_matrix[i][j] = sub_matrix[i - 1][j - 1] + 1
                    max_sub_len = sub_matrix[i][j] if sub_matrix[i][j] > max_sub_len else max_sub_len

    paths = np.sqrt(paths)
    s = paths[l1, l2]
    return s, paths.T, [max_sub_len]


def calculate_attenuate_weight(seqLen1, seqLen2, com_ls):
    weight = 0
    for comlen in com_ls:
        weight = weight + comlen / seqLen1 * comlen / seqLen2
    return 1 - weight

def MaxSubSeries(s1, s2):
    # 取较大的标准差
    sdt = np.std(s1, ddof=1) if np.std(s1, ddof=1) > np.std(s2, ddof=1) else np.std(s2, ddof=1)
    # print("两个序列最大标准差:" + str(sdt))
    l1 = len(s1)
    l2 = len(s2)
    
    sub_matrix = np.full((l1, l2), 0)  # 全部赋予0
    max_sub_len = 0

    
    for i in range(l1):
        for j in range(l2):
            if np.linalg.norm(s1[i] - s2[j], ord=2) < sdt:
                if i == 0 or j == 0:
                    sub_matrix[i][j] = 1
                else:
                    sub_matrix[i][j] = sub_matrix[i - 1][j - 1] + 1
                    max_sub_len = sub_matrix[i][j] if sub_matrix[i][j] > max_sub_len else max_sub_len

    return [max_sub_len]

if __name__ == '__main__':
    pathe=pd.read_csv('608694.csv')
    k=[]
    for index, row in pathe.iterrows():
        k.append(row.tolist())
    # 测试数据
    s1=[]
    for i in range(len(k[0])):
        if k[0][i]!=k[0][i]:
            break
        if i%5==0:
            s1.append((k[0][i],k[1][i]))
    s2=[]
    for i in range(len(k[3])):
        if k[3][i]!=k[3][i]:
            break
        if i%5==0:
            s2.append((k[2][i],k[3][i]))
    s3=[]
    for i in range(len(k[5])):
        if k[5][i]!=k[5][i]:
            break
        if i%5==0:
            s3.append((k[4][i],k[5][i]))
    s1 = np.array(s1)
    s2 = np.array(s2)
    s3 = np.array(s3)

    # 原始算法
    distance12, paths12, max_sub12 = TimeSeriesSimilarityImprove(s1, s2)
    distance13, paths13, max_sub13 = TimeSeriesSimilarityImprove(s1, s3)
    distance23, paths23, max_sub23 = TimeSeriesSimilarityImprove(s2, s3)
    print("更新前s1和s2距离：" + str(distance12))
    print("更新前s1和s3距离：" + str(distance13))
    print("更新前s2和s3距离：" + str(distance23))
    # 衰减系数
    weight12 = calculate_attenuate_weight(len(s1), len(s2), max_sub12)
    weight13 = calculate_attenuate_weight(len(s1), len(s3), max_sub13)
    weight23 = calculate_attenuate_weight(len(s2), len(s3), max_sub23)

    # 更新距离
    print("更新后s1和s2距离：" + str(distance12 * weight12))
    print("更新后s1和s3距离：" + str(distance13 * weight13))
    print("更新后s2和s3距离：" + str(distance23 * weight23))


    ## A cosine is for template; sin and cos are offset by 25 samples
    alignment1 = dtw(s2, s1, keep_internals=True)
    print(alignment1.distance)
    alignment2 = dtw(s3, s1, keep_internals=True)
    print(alignment2.distance)
    alignment3 = dtw(s2, s3, keep_internals=True)
    print(alignment3.distance)
    print("更新后s1和s2距离：" + str(alignment1.distance * weight12))
    print("更新后s1和s3距离：" + str(alignment2.distance * weight13))
    print("更新后s2和s3距离：" + str(alignment2.distance * weight13))
  

