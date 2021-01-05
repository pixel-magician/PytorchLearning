import torch
import math


x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)


# 在这个例子中，输出y是(x, x^2, x^3)的线性函数，所以我们可以把它看作一个线性层神经网络
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# 声明了一个简易版的神经网络模型
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# 声明一个变量保存损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-6
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for t in range(2000):
    # 通过神经网络计算结果
    y_pred = model(xx)
    # 调用损失函数，计算误差
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 清空权重,model.zero_grad()
    optimizer.zero_grad()
    # 反向传播，自动求导
    loss.backward()

    # 更新权重值
    optimizer.step()

linear_layer = model[0]

# bias偏移， weight权重
print(
    f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
