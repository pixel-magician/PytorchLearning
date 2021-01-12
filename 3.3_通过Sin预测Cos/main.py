import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math
import random
import NeuralNet


# 定义超参数
TIME_STEP = 10  # rnn 时序步长数
INPUT_SIZE = 1  # rnn 的输入维度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_SIZE = 64  # of rnn 隐藏单元个数
EPOCHS = 300  # 总共训练次数
h_state = None  # 隐藏层状态

# steps = np.linspace(0, np.pi*2, 256, dtype=np.float32)
# x_np = np.sin(steps)
# y_np = np.cos(steps)


# plt.figure(1)
# plt.suptitle('Sin and Cos',fontsize='18')
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()

rnn = NeuralNet.RNN(INPUT_SIZE, H_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(rnn.parameters())
loss_fn = nn.MSELoss()

rnn.train()
plt.figure(2)
for step in range(EPOCHS):
    start, end = step*np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    x = x.to(DEVICE)
    pred, h_state = rnn(x, h_state)
    h_state = h_state.data
    loss = loss_fn(pred.cpu(), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(step+1) % 20 == 0:
        print("EPOCHS: {},Loss:{:4f}".format(step, loss))
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, pred.cpu().data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.01)

plt.pause(2)
