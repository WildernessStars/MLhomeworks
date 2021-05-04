import numpy as np
import math
# sigmoid
def sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)）
    return 1/(1+math.exp(-x))


def deriv_sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)在x点的梯度）
    temp=1/(1+math.exp(-x))
    return temp*(1-temp)


# loss
def mse_loss(y_true, y_pred):
    # （需要填写的地方，输入真实标记和预测值返回他们的MSE（均方误差）,其中真实标记和预测值都是长度相同的向量）

    E=0
    for y_t, y_p in zip(y_true, y_pred):
        E+=(y_t-y_p)**2
    return E/len(y_true)

class NeuralNetwork_221():
    def __init__(self):
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        # 以上为神经网络中的变量，其中具体含义见网络图

    def predict(self,x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 5000
        last_loss=0;last_loss_ver=100;
        rounds=0
        round2=0
        for epoch in range(epochs):
            for x, y_true in zip(data[0:350], all_y_trues[0:350]):
                # 以下部分为向前传播过程，请完成
                sum_h1 =  self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 =  sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 =  sigmoid(sum_h2)

                sum_ol =  self.w5 * h1 + self.w6 * h2 + self.b3
                ol =  sigmoid(sum_ol)
                y_pred = ol

                # 以下部分为计算梯度，请完成
                d_L_d_ypred =  -2*ol*(1-ol)*(y_true-y_pred)#-2
                # 输出层梯
                d_ypred_d_w5 = learn_rate*d_L_d_ypred*h1
                d_ypred_d_w6 = learn_rate*d_L_d_ypred*h2
                d_ypred_d_b3 = learn_rate*d_L_d_ypred
                d_ypred_d_h1 = h1*(1-h1)*d_L_d_ypred*self.w5
                d_ypred_d_h2 = h2*(1-h2)*d_L_d_ypred*self.w6

                # 隐层梯度
                d_h1_d_w1 = learn_rate*d_ypred_d_h1*x[0]
                d_h1_d_w2 = learn_rate*d_ypred_d_h1*x[1]
                d_h1_d_b1 = learn_rate*d_ypred_d_h1

                d_h2_d_w3 = learn_rate*d_ypred_d_h2*x[0]
                d_h2_d_w4 = learn_rate*d_ypred_d_h2*x[1]
                d_h2_d_b2 = learn_rate*d_ypred_d_h2

                # 更新权重和偏置
                self.w5 -= d_ypred_d_w5
                self.w6 -= d_ypred_d_w6
                self.b3 -= d_ypred_d_b3
                self.w1 -= d_h1_d_w1
                self.w2 -= d_h1_d_w2
                self.b1 -= d_h1_d_b1
                self.w3 -= d_h2_d_w3 
                self.w4 -= d_h2_d_w4
                self.b2 -= d_h2_d_b2
            
            # 计算epoch的loss
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.predict, 1, data[0:350])
                loss = mse_loss(all_y_trues[0:350], y_preds)
                print("Epoch %d loss: %.7f" % (epoch, loss))
                loss_ver = self.early_stop(data[351:],all_y_trues[351:])
                if loss_ver<last_loss_ver:
                    rounds+=1           
                else:
                    rounds=0
                if loss>last_loss:
                    round2+=1
                else:
                    round2=0
                if round2>=3 and rounds>=20:
                    break
                last_loss=loss
                last_loss_ver=loss_ver
                

    def early_stop(self,X_ver,y_true):
        y_ver=[]
        for i in X_ver:
            y_ver.append(self.predict(i))
        loss_ver = mse_loss(y_ver, y_true)
        return loss_ver

def main():
    import numpy as np
    X_train = np.genfromtxt('./data/train_feature.csv', delimiter=',')
    y_train = np.genfromtxt('./data/train_target.csv', delimiter=',')
    X_test = np.genfromtxt('./data/test_feature.csv', delimiter=',')#读取测试样本特征
    
    network = NeuralNetwork_221()
    network.train(X_train, y_train)
    y_pred=[]
    for i in X_test:
        y_pred.append(network.predict(i))#将预测值存入y_pred(list)内
    ##############
    # （需要填写的地方，选定阈值，将输出对率结果转化为预测结果并输出）
    y_train_p=[]
    for i in X_train:
        y_train_p.append(network.predict(i))
    m_t=0;m_f=0;
    for rows in range(len(y_train)): 
        if y_train[rows]==1:
            m_t+=1
        else:
            m_f+=1
    threshold=(m_t/m_f)/(1+m_t/m_f)
    for i in range(len(y_train_p)):
        if y_train_p[i]>=threshold:
            y_train_p[i]=int(1)
        else:
            y_train_p[i]=int(0)
    t=0
    for i in range(len(y_train_p)):
        if y_train_p[i]==y_train[i]:
            t+=1
  #  print(t)
    for i in range(len(y_pred)):
        if y_pred[i]>=threshold:
            y_pred[i]=int(1)
        else:
            y_pred[i]=int(0)
    np.savetxt('171860027_ypred.csv', y_pred, delimiter = ',') 

main()