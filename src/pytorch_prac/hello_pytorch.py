# -*- coding:utf-8 -*-
import torch

a = torch.Tensor([[2, 3], [4, 5]])
print(a)

a = torch.randn((2, 3))

print(a)
print(a[0, 2])

b_numpy = a.numpy()
print(b_numpy)


aa =torch.from_numpy(b_numpy)
print(aa)
# view 拼接成一行
bb = aa.view([-1, 1])
print(bb)

cc = aa.view([1, -1])

print(cc)