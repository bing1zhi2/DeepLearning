# -*- coding:utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

"""
尝试使用pytorch API 对 carf10数据 加载数据,, 对每个批次按当前最宽长度进行pad
"""

data_dir = "D:\dataset\sifar10"


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0 )

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)


# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()
pad = transforms.Pad(padding=10,fill=0)

ss=  torchvision.utils.make_grid(images)
aaa = ss / 2 + 0.5  # unnormalize
batch_imgs=aaa.numpy()

img_list=[]
for i in range(batch_imgs.shape[0]):
    n_img = batch_imgs[i]
    n_img =n_img.astype(np.uint8)
    n_img = np.transpose(n_img,[1,2,0])
    plt.imshow(n_img)
    plt.show()
    images2 = toPIL(n_img)
    pad_img = pad(images2)
    print("pad size", pad_img.size)

    img2_tensor = toTensor(pad_img)
    img_list.append(img2_tensor)

pad_images = torch.stack(img_list)

print(pad_images.shape)


# imshow(torchvision.utils.make_grid(pad_images))


