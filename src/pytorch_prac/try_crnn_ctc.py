# -*- coding:utf-8 -*-
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from src.pytorch_prac.classic_network.crnn import CRNN

"""
模拟使用 crnn 训练不定长图片,   需要对一个批次内的图片做pad,
所以需要支持训练时不同批次间的宽度不同
"""

batchSize=10

imgH=32
imgW=100
nclass = 5990

beta1=0.5

crnn = CRNN(imgH, 1, nclass, imgW)
print(crnn)

criterion = torch.nn.CTCLoss()
optimizer = optim.Adam(crnn.parameters(), lr=0.01,
                       betas=(beta1, 0.999))
C=nclass

# 输入1,  长度为100
input = torch.randn(batchSize, 1, imgH, imgW)
# text = torch.randint()
out = crnn(input)
print(out.shape) # torch.Size([26, 10, 5990])


T= out.shape[0]
N = batchSize
S = 30      # Target sequence length of longest target in batch
S_min = 10  # Minimum target length, for demonstration purposes

target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
loss = criterion(out, target, input_lengths, target_lengths)
print(loss)

#输入宽度2
imgW=280
input2 = torch.randn(batchSize, 1, imgH, imgW)
out2 = crnn(input2)
print("out2 ",out2.shape) # torch.Size([71, 10, 5990])
T = out2.shape[0]
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
loss = criterion(out2, target, input_lengths, target_lengths)
print(loss)

"""
result:

conv size [w, b, c]: torch.Size([26, 10, 512])
torch.Size([26, 10, 5990])
tensor(inf, grad_fn=<MeanBackward0>)

conv size [w, b, c]: torch.Size([71, 10, 512])
out2  torch.Size([71, 10, 5990])
tensor(-2.8770, grad_fn=<MeanBackward0>)
"""