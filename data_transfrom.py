import torchvision
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from test import train_dataset,test_dataset
device = torch.device("cuda")
# train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
#                                          download=True)
# test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
#                                          download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 11)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


train_step = 0
test_step = 0
loss = nn.CrossEntropyLoss()
# loss.to(device)
tudui = torch.load('moudlerec.pth')
# tudui = Tudui()
# tudui.to(device)
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
epoch = 200
for i in range(epoch):
    print("--------- 第{}轮训练开始".format(i+1))
    running_loss = 0.0
    for data in train_dataloader:
        imgs, targets = data

        # imgs = imgs.to(device)
        # targets = targets.to(device)
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
        train_step = train_step + 1
        if train_step % 100 == 0:
            print("训练次数{}，loss{}".format(train_step, running_loss.item()))

    test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # imgs = imgs.to(device)
            # targets = targets.to(device)
            outputs = tudui(imgs)
            running_loss = loss(outputs, targets)
            test_loss = test_loss + running_loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集的loss{}".format(test_loss))
    print("整体测试机的准确率{}".format(total_accuracy/len(train_dataset)))
    test_step = test_step + 1
torch.save(tudui,"moudlerec.pth")


