import torch
import torch.nn as nn
import numpy as np
import LR_Net



# https://github.com/zergtant/pytorch-handbook/blob/master/chapter3/3.1-logistic-regression.ipynb

# 加载数据
data = np.loadtxt("3.1_logistic回归实战/data/german.data-numeric")
# print(data)
# 对数据做归一化处理，TODO?:这一步不知道有什么意义
n, l = data.shape
for j in range(l-1):
    meanVal = np.mean(data[:, j])  # 求均值
    stdVal = np.std(data[:, j])  # 求标准差
    data[:, j] = (data[:, j]-meanVal)/stdVal

# print(data)
# 打乱数据
np.random.shuffle(data)

# 不明白lab为什么要减一
train_data = data[:900, :l-1]
train_lab = data[:900, l-1]-1
test_data = data[900:, :l-1]
test_lab = data[900:, l-1]-1

print(data)


# print(train_data)
# print("-----------------------")
# print(train_lab)

#不明白
def test(pred,lab):
    t=pred.max(-1)[1]==lab
    return torch.mean(t.float())

net=LR_Net.LR()
loss_fn=nn.CrossEntropyLoss()
optm=torch.optim.Adam(net.parameters())
epochs=0

for i in range(epochs):
    net.train()
    x=torch.from_numpy(train_data).float()
    y=torch.from_numpy(train_lab).long()
    y_hat=net(x)
    loss=loss_fn(y_hat,y)
    optm.zero_grad()
    loss.backward()
    optm.step()
    if(i+1)%100==0:
        net.eval()
        test_in=torch.from_numpy(test_data).float()
        test_l=torch.from_numpy(test_lab).long()
        test_out=net(test_in)
        accu=test(test_out,test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i+1,loss.item(),accu))