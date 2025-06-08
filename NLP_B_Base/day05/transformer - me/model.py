from output import *


# 创建编码器解码器对象 --  transform 模型类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        # 初始化各层对象
        self.encoder = encoder      # 编码器层
        self.decoder = decoder      # 解码器层
        self.src_embed = src_embed      # 编码器输入层
        self.tgt_embed = tgt_embed      # 解码器输入层
        self.generator = generator      # 解码器输出层

    def forward(self, src, tgt, src_mask, tgt_mask):

        m = self.encoder(self.src_embed(src), src_mask)             # 生成中间语义张量
        x = self.decoder(self.src_embed(tgt), m, src_mask, tgt_mask)
        output = self.generator(x)

        return output


# 创建transformer模型
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout_p=0.1):
    c = copy.deepcopy

    ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)      # 创建前馈全连接层
    pe = PositionalEncoding(d_model=d_model, dropout_p=dropout_p)       # 创建位置编码子层对象
    attn = MultiHeadedAttention(h, d_model, dropout_p)      # 创建多头注意力子层对象
    el = EncoderLayer(d_model, c(attn), c(ff), dropout_p)           # 创建编码器层对象
    dl = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout_p)          # 创建解码器层对象

    # 创建模型对象
    model = EncoderDecoder(
        Encoder(el, N),     # 编码器	- 由 N 个编码器子层构成
        Decoder(dl, N),     # 解码器	- 由 N 个解码器子层构成
        nn.Sequential(Embeddings(src_vocab, d_model), c(pe)),  # 编码器 的输入
        nn.Sequential(Embeddings(tgt_vocab, d_model), c(pe)),  # 解码器 的输入
        Generator(d_model, tgt_vocab)  # 解码器的输出
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == '__main__':
    # 词表大小
    source_vocab = 1000         # 编码器词表大小
    target_vocab = 1000         # 解码器词表大小

    # 创建transformer模型对象
    model = make_model(source_vocab, target_vocab)

    # 随机生成 输入句子
    source = target = torch.LongTensor([[1, 2, 3, 8], [3, 4, 1, 8]])

    # 下三角掩码矩阵
    source_mask = target_mask = torch.tril(torch.ones(size=(8, 4, 4))).type(torch.uint8)

    # 调用模型得到预测结果
    output = model(source, target, source_mask, target_mask)
    print('output--->', output.shape, output)
