import copy

from encoder import *


# 创建解码器层类
class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_p=0.1):
        super().__init__()
        self.size = size
        self.self_attn = self_attn  # 多头自注意力子层
        self.src_attn = src_attn  # 多头一般注意力子层
        self.feed_forward = feed_forward  # 前馈全连接子层
        # 初始化子层对象
        self.sublayers = clones(SublayerConnection(self.size, dropout_p), 3)  # 克隆三个子层连接网络子层对象

    def forward(self, x, m, padding_mask, casual_mask):
        """
        :param x: 解码器的输入
        :param m: 编码器的输出
        :param padding_mask: 填充掩码
        :param casual_mask: 因果掩码
        :return:
        """
        # 掩码多头自注意力子层
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, casual_mask))
        # 掩码一般注意力子层
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, padding_mask))
        # 前馈全连接子层
        x = self.sublayers[2](x, self.feed_forward)
        return x


# 创建编码器类
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)  # 克隆多个解码器层

    def forward(self, x, m, padding_mask, casual_mask):
        for layer in self.layers:  # 遍历出多个编码器子层
            # 传入参数：解码器输入，编码器输出，填充掩码，因果掩码
            x = layer(x, m, padding_mask, casual_mask)
        return x


def te01_decoder():
    vocab = 1000  # 词表大小是1000
    d_model = 512

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(x)  # [2, 4, 512]

    dropout_p = 0.2  # 置0概率为0.2
    max_len = 60  # 句子最大长度
    my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    pe_result = my_pe(embedded_result)

    # 类的实例化参数与解码器层类似, 相比多出了src_attn, 但是和self_attn是同一个类.
    head = 8
    d_ff = 64
    size = 512
    self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout_p)

    # 前馈全连接层也和之前相同
    my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    # 产生编码器结果
    # 注意此函数返回编码以后的结果 要有返回值, dm_test_Encoder函数后return en_result
    en_result = te03_encoder()
    # 因果掩码
    casual_mask = torch.tril(torch.ones(size=(1, 1, 4, 4))).type(torch.uint8)
    # 填充掩码
    padding_mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)

    c = copy.deepcopy
    # 创建解码器层对象
    my_dl = DecoderLayer(size, c(self_attn), c(src_attn), c(my_ff), dropout_p)

    # 创建解码器对象
    my_decoder = Decoder(my_dl, 2)
    print('my_decoder--->', my_decoder)
    for i in my_decoder.layers:
        i.parameters().requires_grad = False
        print(1111)
    de_result = my_decoder(pe_result, en_result, padding_mask, casual_mask)

    return de_result


if __name__ == '__main__':
    vocab = 1000  # 词表大小是1000
    d_model = 512

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])

    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(x)  # [2, 4, 512]

    dropout_p = 0.2  # 置0概率为0.2
    max_len = 60  # 句子最大长度
    my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    pe_result = my_pe(embedded_result)

    # 类的实例化参数与解码器层类似, 相比多出了src_attn, 但是和self_attn是同一个类.
    head = 8
    d_ff = 64
    size = 512
    self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout_p)

    # 前馈全连接层也和之前相同
    my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    # 产生编码器结果
    # 注意此函数返回编码以后的结果 要有返回值, dm_test_Encoder函数后return en_result
    en_result = te03_encoder()
    # 因果掩码
    casual_mask = torch.tril(torch.ones(size=(1, 1, 4, 4))).type(torch.uint8)
    # 填充掩码
    padding_mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)

    c = copy.deepcopy
    # 创建解码器层对象
    my_dl = DecoderLayer(size, c(self_attn), c(src_attn), c(my_ff), dropout_p)

    # 创建解码器对象
    my_decoder = Decoder(my_dl, 2)  # 解码器与两层解码器子层组成
    de_result = my_decoder(pe_result, en_result, padding_mask, casual_mask)
    # print('de_result --->', de_result)
    # print('de_result.shape --->', de_result.shape)			# torch.Size([2, 4, 512])
