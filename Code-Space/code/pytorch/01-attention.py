# self-attention 实现​
from math import sqrt
import torch
from torch import nn


class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim​
    # q : batch_size * input_dim * dim_k​
    # k : batch_size * input_dim * dim_k​
    # v : batch_size * input_dim * dim_v​
    def __init__(self, input_dim, dim_k, dim_v):
        # super(Self_Attention, self).__init__()
        super().__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k​
        K = self.k(x)  # K: batch_size * seq_len * dim_k​
        V = self.v(x)  # V: batch_size * seq_len * dim_v​
        print(f"Q_Shape:{Q.shape},K_Shape:{K.shape},V_Shape:{V.shape}\n")

        # Q * K.T() #    * seq_len * seq_len​
        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1)) * self._norm_fact)  # ! 这里是`nn.Softmax(dim=-1)()`
        print(f">>>atten:{atten}\n")

        # Q * K.T() * V # batch_size * seq_len * dim_v​
        output = torch.bmm(atten, V)

        return output


X = torch.randn(4, 3, 2)
print(f"X向量矩阵:{X}\n")
self_atten = Self_Attention(2, 4, 5)  # input_dim:2, k_dim:4, v_dim:5​
res = self_atten(X)
print(f">>>res:{res}\n")

print(res.shape)  # [4,3,5]​
print(res.device)

if torch.cuda.is_available():
    print("True")
else:
    print("False")
