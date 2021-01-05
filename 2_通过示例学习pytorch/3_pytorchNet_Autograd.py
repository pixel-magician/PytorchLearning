# 用三阶方程式模拟正弦函数 y = ax^3 + bx^2 + cx + d
import torch
import math

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") #在gpu上运行

# 随机创建输入输出数据
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 随机初始化权重值，三阶方程就是简化的设计网络，a、b、c、d代表神经网络中的权重.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

# 学习率
learning_rate = 1e-6

for t in range(2000):
    # 向前传递，通过神经网络计算结果
    # y = ax^3 + bx^2 + cx + d
    y_pred = a*x**3 + b*x**2 + c*x + d

    # 计算误差
    # np.square(a) 等价于 a**2
    # 最后的sum()表示将数组里所有的loss数据叠加（x, y, y_pred 都是数组哦）,所以要sum求和
    loss = (y_pred-y).pow(2).sum()
    if t % 100 == 99:
        print(t+1, loss.item())

    # 反向传播，内部自动求导
    loss.backward()

    # 反向传播了，更新神经网络的权重
    # torch.no_grad()的方法体的计算不会被算在自动求导范围内
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

    # 清空权重
    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None

print('Result: y = %fx^3 + %fx^2 + %fx + %f' % (a, b, c, d))
print(f'Result: y = {a} x^3 + {b} x^2 + {c} x + {d} ')
