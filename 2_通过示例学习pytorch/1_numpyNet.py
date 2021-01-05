# 用三阶方程式模拟正弦函数 y = ax^3 + bx^2 + cx + d
import numpy as np
import math

# 随机创建输入输出数据
x = np.linspace(-math.pi, math.pi, 2000)  # 取得2000个数
y = np.sin(x)

# 随机初始化权重值，三阶方程就是简化的设计网络，a、b、c、d代表神经网络中的权重.
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

# 学习率
learing_rate = 1e-6

for t in range(2000):
    # 向前传递，通过神经网络计算结果
    # y = ax^3 + bx^2 + cx + d
    y_pred = a*x**3 + b*x**2 + c*x + d

    # 计算误差
    # np.square(a) 等价于 a**2
    # 最后的sum()表示将数组里所有的loss数据叠加（x, y, y_pred 都是数组哦）,所以要sum求和
    loss = np.square(y_pred-y).sum()
    if t % 100 == 99:
        print(t+1, loss)

    # 原文注释：Backprop to compute gradients of a, b, c, d with respect to loss
    # 偏导数，梯度这部分数学知识依然没搞明白，明明大学里还学的不错，现在一点也想不起来了
    grad_y_pred = 2.0*(y_pred-y)
    grad_d = grad_y_pred.sum()
    grad_c = (grad_y_pred*x).sum()
    grad_b = (grad_y_pred*x**2).sum()
    grad_a = (grad_y_pred*x**3).sum()

    # 反向传播了，更新神经网络的权重
    a -= learing_rate * grad_a
    b -= learing_rate * grad_b
    c -= learing_rate * grad_c
    d -= learing_rate * grad_d

print('Result: y = %fx^3 + %fx^2 + %fx + %f' % (a, b, c, d))
print(f'Result: y = {a} x^3 + {b} x^2 + {c} x + {d} ')
