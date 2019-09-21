# -*- coding:utf-8 -*-
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


def try_resnet():
    # download and load the pretrained ResNet10
    resnet18 = torchvision.models.resnet18(pretrained=True)

    # 如果只finetune 最后一层，设置：
    for param in resnet18.parameters():
        param.requires_grad = False

    # replace the top layer for finetuning
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 100)

    # forward pass
    images = torch.randn(64, 3, 244, 244)
    outputs = resnet18(images)
    print(outputs.size())

def try_densenet():
    densenet = torchvision.models.densenet121(pretrained=True)
    print(densenet)

