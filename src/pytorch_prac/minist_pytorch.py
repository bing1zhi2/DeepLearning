# -*- coding:utf-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class MinistNet(nn.Module):
    def __init__(self):
        super(MinistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  # in_channels, out_channels, kernel_size, stride
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # input (64, 1, 28, 28)

        x = F.relu(self.conv1(x))
        # (64, 20, 24,24)
        x = F.max_pool2d(x, (2, 2))
        # (64, 20, 12, 12)
        x = F.relu(self.conv2(x))  # (64, 50, 8, 8)
        x = F.max_pool2d(x, (2, 2))  # (64, 50, 4, 4)
        x = x.view(-1, 4 * 4 * 50)  # (64, 800)
        x = F.relu(self.fc1(x))  # (64, 500)
        x = self.fc2(x)  # (64, 10)

        return F.log_softmax(x)


def train(model, train_loader, epoch, optimizer, device):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))


def test(model, device, test_loader):
    test_loss = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            test_loss += F.nll_loss(out, target, reduction="sum").item()
            pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = test_loss / len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        )),
        batch_size=64, shuffle=True, **kwargs
    )
    # # 显示下数据格式
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx, data.size(), target.size())
        if batch_idx == 10:
            break
    # # 0 torch.Size([64, 1, 28, 28]) torch.Size([64])
    # # 1 torch.Size([64, 1, 28, 28]) torch.Size([64])
    # # 2 torch.Size([64, 1, 28, 28]) torch.Size([64])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True, **kwargs
    )

    model = MinistNet().to(device)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, 2):
        train(model, train_loader, epoch, optimizer, device)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()
