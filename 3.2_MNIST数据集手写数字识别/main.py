import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import NeuralNet as Net

# 超参数定义
BATCH_SIZE = 512
EPOCHS = 20  # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("3.2_MNIST数据集手写数字识别/data",
                   train=True,
                   download=True,
                   transform=transforms.Compose(
                       [transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=BATCH_SIZE,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("3.2_MNIST数据集手写数字识别/data",
                   train=False,
                   transform=transforms.Compose(
                       [transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=BATCH_SIZE,
    shuffle=True)


def train(net, device, train_loader, optim, epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optim.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optim.step()
        if(batch_idx+1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(net, device, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


net = Net.Net().to(DEVICE)
optim = optim.Adam(net.parameters())
for epoch in range(1, EPOCHS+1):
    train(net, DEVICE, train_loader, optim, epoch)
    test(net, DEVICE, test_loader)
