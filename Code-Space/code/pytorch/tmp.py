from math import sqrt
from turtle import forward
import torch
from torch import nn


class Self_Attention(nn.Module):
    def __init__(self, input_dim, k_dim, v_dim):
        super().__init__()
        self.q = nn.Linear(input_dim, k_dim)
        self.k = nn.Linear(input_dim, k_dim)
        self.v = nn.Linear(input_dim, v_dim)
        self.norm = 1 / sqrt(k_dim)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1)) * self.norm)
        output = torch.bmm(atten, V)
        return output


X = torch.randn(4, 3, 3)
self_atten = Self_Attention(3, 4, 5)
res = self_atten(X)
print(res.shape)
print(res.device)
