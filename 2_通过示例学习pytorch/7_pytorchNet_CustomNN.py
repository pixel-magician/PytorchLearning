import torch
import math


class Polynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a * x**3 + self.b * x**2 + self.c * x + self.d

    def string(self):
        return 'Result: y = %fx^3 + %fx^2 + %fx + %f' % (self.a, self.b, self.c, self. d)


x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)


net = Polynomial3()

loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)

for t in range(2000):
    y_pred = net(x)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t+1, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {net.string()}')
