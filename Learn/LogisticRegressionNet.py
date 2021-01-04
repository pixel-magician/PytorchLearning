import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionNet(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionNet, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.nn.functional.sigmoid(self.linear(x))
        return y_pred


def Training(model, x_data, y_data, traiining_Count=100):
    LossFunction = torch.nn.BCELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(traiining_Count):
        y_pred = model(x_data)
        loss = LossFunction(y_pred, y_data)
        print("第%d次训练，损失值：%.2f" % (i, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("权重W：%.2f, b：%.2f" %
          (model.linear.weight.item(), model.linear.bias.item()))


if __name__ == "__main__":
    model = LogisticRegressionNet()
    model.cuda()
    x_data = x_data.cuda()
    y_data = y_data.cuda()
    Training(model, x_data, y_data, 1000)
    device = torch.device('cuda')
    test = model(torch.tensor([[4.0]], device=device))
    print("预测结果：%.2f" % test)
    test = model(torch.tensor([[5.0]], device=device))
    print("预测结果：%.2f" % test)
    test = model(torch.tensor([[2.0]], device=device))
    print("预测结果：%.2f" % test)
