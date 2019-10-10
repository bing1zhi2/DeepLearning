import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def try_usr_cifar10():
    data_dir = "D:\dataset\sifar10"


    transform = transforms.Compose(
        [transforms.Scale(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)

    print("=> creating model '{}'".format(""))
    model = models.__dict__["densenet121"]()
    print(model)

    # model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                    momentum=0.9,
                                    weight_decay=1e-4)

    for epoch in range(1):
        running_loss = 0

        for i,data in enumerate(trainloader):
            inputs, targets = data

            # targets = targets.cuda(async=True)
            input_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(targets)

            output = model(input_var)
            loss = criterion(output, target_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(running_loss)
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
def try_size():
    # Download and load the pretrained ResNet-18.
    resnet = torchvision.models.resnet18(pretrained=False)
    densenet = torchvision.models.densenet121(pretrained=True)
    print(densenet)

    # If you want to finetune only the top layer of the model, set as below.
    for param in densenet.parameters():
        param.requires_grad = False

    a = densenet.classifier.in_features
    print(a)
    # Replace the top layer for finetuning.
    # densenet = nn.Linear(densenet.classifier.in_features, 100)  # 100 is an example.

    # Forward pass.
    images = torch.randn(64, 3, 244, 244)
    outputs = densenet(images)
    print(outputs.size())  # (64, 100)

try_size()