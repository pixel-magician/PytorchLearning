import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 160, 5)
        self.fc1 = nn.Linear(160 * 5 * 5, 1200)
        self.fc2 = nn.Linear(1200, 840)
        self.fc3 = nn.Linear(840, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 160 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # 除去批处理维度的其他所有维度
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()





if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Net()
    net = net.to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root='./1_60分钟的闪电战/data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./1_60分钟的闪电战/data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 输出图像的函数

    # 随机获取训练图片
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # 显示图片
    # imshow(torchvision.utils.make_grid(images))
    # 打印图片标签
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    path="./Learn/netData_full"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(),path)

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = dataiter.next()
    # images, labels = dataiter.next()

    # 输出图片
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(4)))

    # images = images.to(device)
    net.cpu()
    net.load_state_dict(torch.load(path))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    # predicted.to(torch.device("cpu"))

    print('Predicted: ', ' '.join('%5s' %
                                  classes[predicted[j]] for j in range(4)))
