import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearNet(torch.nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def Training(model, x_data, y_data, traiining_Count=100):
    LossFunction = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

    for i in range(traiining_Count):
        y_pred = model(x_data)
        loss = LossFunction(y_pred, y_data)
        print("第%d次训练，损失值：%.2f" %(i, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("权重W：%.2f, b：%.2f" %(model.linear.weight.item(), model.linear.bias.item()))


if __name__ == "__main__":
    model = LinearNet()
    Training(model, x_data, y_data, 1000)
    test=model(torch.tensor([[4.0]]))
    print("预测结果：%.2f"%test)
