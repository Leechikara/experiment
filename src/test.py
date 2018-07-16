# coding = utf-8
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

a = nn.Embedding(3,3,padding_idx=0)
b = torch.LongTensor([1,2])
c = a(b)

print(c)
print(c.sum(dim=1))