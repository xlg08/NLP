'''
    在 transfomer 中，编码器与解码器，
        克隆函数、子层连接(残差连接、层归一化)、多头注意力机制、前馈全连接层
    是公共的网络子层，因此抽取为一个公共的模块
'''
import torch
import math
from torch import nn
import copy


# ############################################ 克隆方法 ##############################################
def clones(module, N):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


# ############################################ 多头注意力机制 ###############################################
# 多头注意力计算出动态张量 C
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]

    # 权重分数矩阵计算
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, value=-1e9)

    p_attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 计算出动态张量 C
    c = torch.matmul(p_attn, value)

    return c, p_attn


# 创建多头注意力机制类，多头注意力神经网络子层
class MultiHeadedAttention(nn.Module):

    def __init__(self, head, d_model, dropout_p=0.1):
        super().__init__()

        assert d_model % head == 0, 'd_model不能被head整数'
        self.d_k = d_model // head
        self.dropout = nn.Dropout(p=dropout_p)
        self.head = head
        self.attn = None

        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size()[0]
        output_list = []

        for model, x in zip(self.linears, (query, key, value)):
            output = model(x)
            output = output.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            output_list.append(output)

        query = output_list[0]
        key = output_list[1]
        value = output_list[2]

        x, p_attn = attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        x = self.linears[-1](x)

        return x


# ####################################### 前馈全连接子层 #######################################
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        output = torch.relu(self.linear1(x))
        output = self.dropout(output)
        return self.linear2(output)


# ####################################### 子层连接(残差链接+层归一化)子层 #######################################
# ####################################### 子层连接(层归一化)子层 #######################################
# 层归一化层
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))

    def forward(self, x):

        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        x = self.w * ((x - mean) / (std + self.eps)) + self.b
        return x


# ####################################### 子层连接(残差连接)子层 #######################################
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout_p=0.1):
        super().__init__()
        self.size = size
        self.norm = LayerNorm(self.size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, sublayer):
        x = self.norm(x + self.dropout(sublayer(x)))
        return x






