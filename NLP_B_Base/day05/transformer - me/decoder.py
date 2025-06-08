
from encoder import *


class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_p=0.1):
        super().__init__()
        self.size = size
        self.self_attn = self_attn  # 多头自注意力子层
        self.src_attn = src_attn  # 多头一般注意力子层
        self.feed_forward = feed_forward  # 前馈全连接子层
        # 初始化子层连接层对象
        self.sublayers = clones(SublayerConnection(self.size, dropout_p), 3)  # 解码器有三个网络子层，因此拷贝三个子层连接

    def forward(self, x, m, padding_mask, casual_mask):

        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, casual_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, padding_mask))
        x = self.sublayers[2](x, self.feed_forward)

        return x


# 创建编码器类
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)

    def forward(self, x, m, padding_mask, casual_mask):
        for layer in self.layers:
            x = layer(x, m, padding_mask, casual_mask)
        return x

