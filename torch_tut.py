#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch tutorial

Created on Tue Apr 21 09:52:12 2020

@author: gp
"""
import torch
cuda = torch.device('cuda:0')

# tensor basics
x = torch.empty(5,3)
print(x)

x=torch.rand(5,3)
print(x)
'''
x = torch.rand(5,3, dtype = torch.long)
print(x)
'''
x = x.new_ones(5,3, dtype=torch.int32)
print(x)

x = torch.rand_like(x, dtype=torch.float32)
print(x)

print(x.size())
print(torch.Size([5,3]))

a = torch.tensor([3,2], dtype=torch.int32)
b = torch.tensor([6,7], dtype=torch.int32)
print(a+b)
print(torch.add(a,b))

x = torch.rand(5,5, dtype=torch.float32, device=cuda)
print(x.view(-1,25))
y = x.view(1,25)
print(y)

a = x[:,2:4]
print(a)

p = a.clone()
print(a.shape)
b = p.view(-1, 5)
print(b)

# autograd

x = torch.tensor([2, 3], dtype=torch.float32, device=cuda, requires_grad=True)
print(x)
y = x*x+4*x+3
print(y)
print(x)
print(y.grad_fn)
z = y*2+3
out = z.mean()
print(out)
out.backward()
print(out)
print(x.grad)
print(y.grad)
print(z.grad)
