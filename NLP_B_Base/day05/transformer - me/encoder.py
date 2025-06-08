from input import *
from commonsLayer import *

# 创建编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forwad, dropout_p=0.1):
        super().__init__()
        self.size = size  # 词向量维度
        self.self_attn = self_attn  # 多头自注意力机制对象
        self.feed_forwad = feed_forwad  # 前馈全连接对象
        self.dropout_p = dropout_p
        self.sublayer = clones(SublayerConnection(self.size, self.dropout_p), 2)        # 编码器有两个网络子层，因此拷贝两个子层连接

    def forward(self, x, mask):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forwad)

        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return x

